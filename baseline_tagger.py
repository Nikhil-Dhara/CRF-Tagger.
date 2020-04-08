import hw2_corpus_tool
import pycrfsuite
import time
import sys


class Tagger:
    def __init__(self):
        # ram_train_path='/Users/nikhildhara/Desktop/nlp-tagger/ram_data/train'
        # ram_test_path='/Users/nikhildhara/Desktop/nlp-tagger/ram_data/dev'

        # self.train_set_path = '/Users/nikhildhara/Desktop/nlp-tagger/ram_data/train'
        # self.test_set_path = '/Users/nikhildhara/Desktop/nlp-tagger/ram_data/dev'

        self.train_set_path = sys.argv[1]
        self.test_set_path = sys.argv[2]
        self.output_file = sys.argv[3]
        self.trainlabels = []
        self.testlabels = []
        self.trainfeatures = []
        self.testfeatures = []
        self.training_data = []
        self.testing_data = []
        self.trainer = None

    def train(self):
        self.training_data = list(hw2_corpus_tool.get_data(self.train_set_path))
        for utterance in self.training_data:
            count = 0
            utterance_label = []
            current_speaker = ''
            past_speaker = ''
            feature_list = []
            for val in utterance:
                line_features = []
                current_speaker = getattr(val, "speaker")
                if count == 0:
                    line_features.append('fs')
                if count > 0:
                    if current_speaker != past_speaker:
                        line_features.append('ch')
                tokens_tags = getattr(val, "pos")
                if tokens_tags is not None:
                    for x in tokens_tags:
                        line_features.append('TOKEN_' + getattr(x, 'token'))
                    for x in tokens_tags:
                        line_features.append('POS_' + getattr(x, 'pos'))
                utterance_label.append(getattr(val, "act_tag"))
                feature_list.append(line_features)
                count += 1
                past_speaker = current_speaker
            self.trainfeatures.append(feature_list)
            self.trainlabels.append(utterance_label)

    def build_model(self):
        self.trainer = pycrfsuite.Trainer(verbose=False)
        for x, y in zip(self.trainfeatures, self.trainlabels):
            self.trainer.append(x, y)
        self.trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-2,  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        self.trainer.train('tagger_model_trained')
        print(self.trainer)

    def test(self):
        self.testing_data = list(hw2_corpus_tool.get_data(self.test_set_path))
        for utterance in self.testing_data:

            count = 0
            utterance_label = []
            past_speaker = ''
            feature_list = []
            for val in utterance:
                line_features = []
                current_speaker = getattr(val, "speaker")
                if count == 0:
                    line_features.append('fs')
                if count > 0:
                    if current_speaker != past_speaker:
                        line_features.append('ch')
                tokens_tags = getattr(val, "pos")
                if tokens_tags is not None:
                    for x in tokens_tags:
                        line_features.append('TOKEN_' + getattr(x, 'token'))
                    for x in tokens_tags:
                        line_features.append('POS_' + getattr(x, 'pos'))
                utterance_label.append(getattr(val, 'act_tag'))
                feature_list.append(line_features)
                count += 1
                past_speaker = current_speaker
            self.testfeatures.append(feature_list)
            self.testlabels.append(utterance_label)

    def check_accuracy(self):
        tagger = pycrfsuite.Tagger()
        tagger.open('tagger_model_trained')
        sum = 0
        cor = 0
        for i in range(len(self.testfeatures)):
            pred = tagger.tag(self.testfeatures[i])
            for j in range(len(pred)):
                if pred[j] == self.testlabels[i][j]:
                    cor += 1
                sum += 1
        print("Baseline Acccuracy = " + str(cor * 100 / sum) + '%\n')

    def write_output(self):
        tagger = pycrfsuite.Tagger()
        tagger.open('tagger_model_trained')
        f = open(self.output_file, 'w')
        for i in range(len(self.testfeatures)):
            pred = tagger.tag(self.testfeatures[i])
            for j in range(len(pred)):
                f.write(pred[j] + '\n')
            f.write('\n')
        f.close()


if __name__ == '__main__':
    # start_time = time.time()
    tag = Tagger()
    tag.train()
    print(' Input Features Loaded')
    tag.build_model()
    print('CRF Model trained on input data')
    tag.test()
    print('CRF Model tested on testing data')
    #tag.check_accuracy()
    tag.write_output()
    # end_time=time.time()
    # print('Time Taken',end_time-start_time)
