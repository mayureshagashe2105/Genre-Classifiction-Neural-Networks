import numpy as np
import pandas as pd
import scipy.special

data = pd.read_csv('songDb.csv', sep=',')
print(data.head())
print(data.info())

data['time_signature'] = pd.to_numeric(data['time_signature'], errors='coerce')
data['Genre'].replace(np.NaN, '-1', inplace=True)
data['time_signature'] = data['time_signature'].replace(np.nan, data['time_signature'].mean())
data['Tempo'] = pd.to_numeric(data['Tempo'], errors='coerce')
data['Tempo'].replace(np.nan, data['Tempo'].mean(), inplace=True)
print(data.info())

'''for i in data.columns:
    print(data[i].isnull().value_counts())'''

X = np.array(data.drop(['Name', 'Type', 'ID', 'Uri', 'Ref_Track', 'URL_features', 'Genre', 'Key', 'Mode',
                        'time_signature', 'Duration_ms'], axis=1))
print(X.shape[1], 'inputs')
y = np.array(data['Genre'])
print(len(np.unique(y)), 'before grouping')

print(len(y), 'Total')


def label_encoder(arr):
    for i in range(len(arr)):
        if str(arr[i]).find('classic') != -1 or str(arr[i]).find('medieval') != -1:
            arr[i] = 0
        elif str(arr[i]).find('folk') != -1:
            arr[i] = 1
        elif str(arr[i]).find('country') != -1:
            arr[i] = 2
        elif str(arr[i]).find('rock') != -1 or str(arr[i]).find('metal') != -1 or str(arr[i]).find('punk') != -1:
            arr[i] = 3
        elif str(arr[i]).find('blues') != -1 or str(arr[i]).find('Blues') != -1 or str(arr[i]).find('hip hop') != -1 or \
                str(arr[i]).find('hiphop') != -1 or str(arr[i]).find('jazz') != -1 or str(arr[i]).find('funk') != -1 or \
                str(arr[i]).find('soul') != -1 or str(arr[i]).find('doo-wop') != -1 or str(arr[i]).find('rap') != -1 or \
                str(arr[i]).find('disco') != -1 or str(arr[i]).find('r&b') != -1:
            arr[i] = 4
        elif str(arr[i]).find('reggae') != -1:
            arr[i] = 5
        elif str(arr[i]).find('pop') != -1 or str(arr[i]).find('dance') != -1 or str(arr[i]).find('scratch') != -1:
            arr[i] = 6
        elif str(arr[i]).find('adult standards') != -1:
            arr[i] = 7
        elif str(arr[i]).find('indie') != -1:
            arr[i] = 7
        elif str(arr[i]).find('edm') != -1 or str(arr[i]).find('ambient') != -1 or \
                str(arr[i]).find('downtempo') != -1 or str(arr[i]).find('electro') != -1 or \
                str(arr[i]).find('house') != -1 or str(arr[i]).find('trance') != -1 or\
                str(arr[i]).find('techno') != -1 or str(arr[i]).find('wave') != -1:
            arr[i] = 8
        elif str(arr[i]).find('romantic') != -1:
            arr[i] = 9
        elif str(arr[i]).find('core') != -1:
            arr[i] = 10
        elif str(arr[i]).find('choir') != -1:
            arr[i] = 11
        elif str(arr[i]).find('beat') != -1:
            arr[i] = 12
        elif str(arr[i]).find('meditation') != -1:
            arr[i] = 13
        else:
            arr[i] = 14
    return arr


y = label_encoder(y)
print(len(np.unique(y)), 'No of output grps after grouping')
yr, fre = np.unique(y, return_counts=True)
print(np.sum(fre), 'Total after grouping')
print(yr)


def StandardScalar(arr):
    try:
        for i in range(arr.shape[1]):
            mean = arr[:, i].mean()
            std = arr[:, i].std()
            arr[:, i] = (arr[:, i] - mean) / std

    except IndexError:
        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean) / std

    finally:
        return arr


X = StandardScalar(X)


def train_test_split(X, y, testing_size=0.2):
    total_no_rows = X.shape[0]
    test_no_rows = int(total_no_rows * testing_size)
    rand_row_no = np.random.randint(0, total_no_rows, test_no_rows)

    X_test = np.array([X[i] for i in rand_row_no])
    X_train = np.delete(X, rand_row_no, axis=0)

    y_test = np.array([y[i] for i in rand_row_no])
    y_train = np.delete(y, rand_row_no, axis=0)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X, y, testing_size=0.2)


class NeuralNetworks:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, epochs, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.epochs = epochs
        self.lr = learning_rate
        self.wih = np.random.randn(self.hnodes, self.inodes) - 0.5
        self.who = np.random.randn(self.onodes, self.hnodes) - 0.5
        #self.hb = np.zeros((self.hnodes, 1))
        #self.ob = np.zeros((self.onodes, 1))
        self.activation_function_ReLu = lambda x: scipy.special.expit(x)

    def train(self, inputs_train, targets_train):
        inputs_train = (inputs_train / 255 * 0.99) + 0.01
        for i in range(self.epochs):
            for j in range(len(targets_train)):
                sample_input = inputs_train[j].reshape(-1, 1)

                hidden_inputs = self.wih.dot(sample_input) #+ self.hb
                hidden_outputs = self.activation_function_ReLu(hidden_inputs)

                outer_inputs = self.who.dot(hidden_outputs) #self.ob
                outer_outputs = self.activation_function_ReLu(outer_inputs)

                target_arr = np.zeros((self.onodes, 1)) + 0.01
                target_arr[targets_train[j]] = 0.99

                self.output_error = target_arr - outer_outputs
                self.hidden_error = self.who.transpose().dot(self.output_error)

                self.who += self.lr * ((self.output_error * outer_outputs * (1 - outer_outputs)).dot(hidden_outputs.transpose()))
                #self.ob += self.lr * (self.output_error * outer_outputs * (1 - outer_outputs))

                self.wih += self.lr * ((self.hidden_error * hidden_outputs * (1 - hidden_outputs)).dot(sample_input.transpose()))
                #self.hb += self.lr * (self.hidden_error * hidden_outputs * (1 - hidden_outputs))
                print(j)
            print(i)

    def predict(self, sample_input):
        sample_input = (sample_input / 255 * 0.99) + 0.01
        sample_input.reshape(-1, 1)

        hidden_inputs = self.wih.dot(sample_input)  # + self.hb
        hidden_outputs = self.activation_function_ReLu(hidden_inputs)

        outer_inputs = self.who.dot(hidden_outputs)  # self.ob
        outer_outputs = self.activation_function_ReLu(outer_inputs)

        return outer_outputs

    def score(self, X_test, y_test):
        counter = 0
        for i in range(len(y_test)):
            output = list(self.predict(X_test[i]))
            if output.index(max(output)) == y_test[i]:
                counter += 1

        return counter / len(y_test) * 100


NN = NeuralNetworks(9, 500, 15, 1, 0.35)
NN.train(X_train, y_train)
acc = NN.score(X_test, y_test)
print(acc)