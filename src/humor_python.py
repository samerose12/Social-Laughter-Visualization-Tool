import numpy as np
import pandas as pd

class Sim:
    def __init__(self, audience_size, has_comedian_column, mediation, comedian, placement_type):
        comedian = comedian.lower()

        #read in jokes from text file
        if has_comedian_column == 'Multiple':
            jokes = pd.read_csv('matrix.txt', header=None, names=['comedian', 'types', 'butt', 'intensity', 'crescendo_count'])
        else:
            jokes = pd.read_csv('matrix.txt', header=None, names=['types', 'butt', 'intensity', 'crescendo_count'])
            jokes.insert(loc=0, column='comedian', value=comedian)


        #make jokes all lowercase
        jokes = jokes.applymap(lambda s:s.lower() if type(s) == str else s)
        self.people = list(set(jokes['comedian'].tolist() + jokes['butt'].dropna().tolist()))


        sqrt_as = int(np.sqrt(audience_size))
        if placement_type.lower() == 'polygon':
            self.make_polygon(self.people, comedian, sqrt_as/2)
        elif placement_type.lower() == 'flyting':
            self.make_polygon(self.people, None, sqrt_as/2)
        elif placement_type.lower() == 'random':
            self.pos_people = sqrt_as/4 + np.random.rand(len(self.people), 2) * sqrt_as/2



        self.jokes = jokes

        self.a_log = np.log10(audience_size)

        self.mediation = mediation
        #this is a CASE-SENSITIVE comparison
        if mediation == 'Face To Face':
            self.pos_audience = self.create_groups(sqrt_as)
        else: #make random points
            self.pos_audience = np.random.rand(audience_size, 2) * sqrt_as

        self.a_mean = [self.pos_audience[:, 0].mean(), self.pos_audience[:, 1].mean()]
        self.a_sqrt = sqrt_as

        self.joke_types = ['dyadic', 'triadic', 'ironic', 'parody', 'satire', 'self-deprecating', 'irony', 'satiric']
        self.mediation_types = {'Face To Face': 1.0, 'Single': 0.10, 'Laugh Track': 0.80, 'Mediated Face to Face': 1.25}
        self.crescendo_effect = {0 : 1, 1 : 1.05, 2 : 1.10, 3 : 1.15}


        self.left_out = np.empty((audience_size, 2))

        self.pos_cur_com = []
        self.pos_cur_butt = []





    def set_pos_people(self, x, y):
        self.pos_people = np.c_[x, y]

    def make_polygon(self, people, com, dist):
        n = len(self.people)
        self.pos_people = np.zeros((n, 2))
        if com != None:
            index_of_com = self.people.index(com)
            self.pos_people[index_of_com] = [dist, dist]
            offset = 1
        else:
            index_of_com = -1
            offset = 0

        top = [np.linalg.norm([0, dist/2]), np.arctan2(dist/2, 0)]
        count = 0
        for i in range(0, n):
            if i == index_of_com:
                continue
            theta = top[1] + ((count) * (2*np.pi / (n-offset)))
            self.pos_people[i] = [dist + top[0]*np.cos(theta), dist + top[0]*np.sin(theta)]
            count += 1


    def create_groups(self, a_sqrt, max_group_size=10):
        a_s = a_sqrt ** 2
        aud = np.zeros((int(a_s), 2))
        index = 0
        #pick the number of randomly sized groups to partition the audience size into
        group_sizes = []
        while a_s > 0:                                         #  1    2     3    4     5     6    7     8     9     10
            r = np.random.choice(range(1, max_group_size+1), p=[0.01, 0.25, 0.25, 0.2, 0.15, 0.05, 0.04, 0.03, 0.01, 0.01]) #less likely to go alone, more likely to go with a few friends, less likely again to go with 8-10 probably
            a_s -= r
            if a_s < 0:
                group_sizes.append(a_s + r)
                a_s = 0
                break
            else:
                group_sizes.append(r)

        #for each group pick a random center point and populate around it
        for size in group_sizes:
            center = a_sqrt/16 + np.random.rand(1, 2) * 7*a_sqrt/8
            for _ in range(size):
                aud[index] = center + (np.random.rand(1,2) - 0.5) * a_sqrt/16
                index += 1

        return aud


    def update_wrt_com(self, index_of_comedian, intensity, index_of_butt):
        c = self.pos_people[index_of_comedian]
        c1 = c + self.normalize(self.a_mean - c)*intensity
        self.pos_people[index_of_comedian] = c1 - self.normalize(self.pos_people[index_of_butt] - c1)*intensity if index_of_butt != None else c1

    def update_wrt_butt(self, index_of_butt, index_of_comedian, intensity):
        if index_of_butt == None:
            return
        b = self.pos_people[index_of_butt]
        b1 = b - self.normalize(self.a_mean - b)*intensity
        self.pos_people[index_of_butt] = b1 - self.normalize(self.pos_people[index_of_comedian] - b1) * intensity


    def normalize(self, v):
        try:
            norm = np.linalg.norm(v, axis=1).reshape(-1, 1)
        except:
            norm = np.linalg.norm(v)
        return v / norm if (norm == 0).all() == False else np.zeros(v.shape)



    def move_audience(self, index_of_butt, index_of_comedian, intensity):
        a = self.pos_audience
        c = self.pos_people[index_of_comedian]
        a1 = a + self.normalize(c - a)*intensity
        a2 = a1 + self.normalize(self.a_mean - a1)*intensity if a.shape[0] > 1 else a1
        b = self.pos_people[index_of_butt] if index_of_butt != None else a2
        self.pos_audience = a2 - self.normalize(b - a2) * intensity




    def step(self, i, intensity_mod):

        self.a_mean = [self.pos_audience[:, 0].mean(), self.pos_audience[:, 1].mean()]
        self.left_out = np.array([-1, -1]).reshape(-1, 2, 1)
        index_of_comedian = self.people.index(self.jokes['comedian'][i])

        try:
            index_of_butt = self.people.index(self.jokes['butt'][i])
        except:
            index_of_butt = None

        mediation_percent = self.mediation_types[self.mediation]

        try:
            crescendo_percent = self.crescendo_effect[self.jokes['crescendo_count'][i]]
        except KeyError:
            print("Only known crescendo counts are 0, 1, 2, 3")
            raise

        try:
            index_of_joke = self.joke_types.index(self.jokes['types'][i])
        except KeyError:
            print("Only known joke types are 'dyadic', 'triadic', 'ironic', 'parody', 'satire'")
            raise

        # handles self-deprecating jokes as dyadic
        if self.jokes['comedian'][i] == self.jokes['butt'][i]:
            index_of_joke = 0

        #calculate intensity
        intensity = self.jokes['intensity'][i] * mediation_percent * crescendo_percent * self.a_sqrt * intensity_mod


        #look at the joke type
        if index_of_joke == 0 or index_of_joke == 5:
            '''Dyadic: increases affiliation between comedian and audience'''
            self.update_wrt_com(index_of_comedian, intensity, index_of_butt)
            self.move_audience(index_of_butt, index_of_comedian, intensity)


        elif index_of_joke == 1:
            '''Triadic: increases affiliation between comedian and audience, and decreases affiliation with the butt'''
            self.update_wrt_butt(index_of_butt, index_of_comedian, intensity)
            self.update_wrt_com(index_of_comedian, intensity, index_of_butt)
            self.move_audience(index_of_butt, index_of_comedian, intensity)



        elif index_of_joke == 2 or index_of_joke == 6:
            '''Ironic: triadic
                - 10% have no affiliative effects'''
            weight = 0.10
            anti_inten = 0
            weighted_intensity = np.random.choice([intensity, anti_inten*intensity], size=(len(self.pos_audience), 1), p=[1-weight, weight])
            self.update_wrt_com(index_of_comedian, intensity, index_of_butt)
            self.update_wrt_butt(index_of_butt, index_of_comedian, intensity)
            self.move_audience(index_of_butt, index_of_comedian, weighted_intensity)

            #lets color the people who don't get the joke
            temp = weighted_intensity
            temp[temp != anti_inten] = -1
            temp[temp == anti_inten] = 1
            temp[temp == -1] = 0
            temp = self.pos_audience * temp
            self.left_out = temp[temp != [0, 0]].reshape(-1, 2, 1)


        elif index_of_joke == 3:
            '''Parody: triadic'''
            self.update_wrt_butt(index_of_butt, index_of_comedian, intensity)
            self.update_wrt_com(index_of_comedian, intensity, index_of_butt)
            self.move_audience(index_of_butt, index_of_comedian, intensity)


        elif index_of_joke == 4 or index_of_joke == 7:
            '''Satire: triadic
                - 25% have half affiliative effects'''
            weight = 0.25
            anti_inten = 0.5
            weighted_intensity = np.random.choice([intensity, anti_inten*intensity], size=(len(self.pos_audience), 1), p=[1-weight, weight])
            self.update_wrt_com(index_of_comedian, intensity, index_of_butt)
            self.update_wrt_butt(index_of_butt, index_of_comedian, intensity)
            self.move_audience(index_of_butt, index_of_comedian, weighted_intensity)

            temp = weighted_intensity
            temp[temp != anti_inten*intensity] = -1
            temp[temp == anti_inten*intensity] = 1
            temp[temp == -1] = 0
            temp = self.pos_audience * temp
            self.left_out = temp[temp != [0, 0]].reshape(-1, 2, 1)


        self.pos_cur_com = self.pos_people[index_of_comedian]
        self.pos_cur_butt = self.pos_people[index_of_butt] if index_of_butt != None else self.pos_cur_com
