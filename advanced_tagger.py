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

    def get_bigram(self, arr):
        return zip(arr, arr[1:])

    def get_trigram(self, arr):
        return zip(arr, arr[1:], arr[2:])

    def train(self):
        self.training_data = list(hw2_corpus_tool.get_data(self.train_set_path))
        for utterance in self.training_data:
            count = 0
            utterance_label = []
            current_speaker = ''
            past_speaker = ''
            feature_list = []
            for val in utterance:
                all_token = []
                all_pos = []
                line_features = []
                current_speaker = val.speaker
                if count == 0:
                    line_features.append('fs')
                if count > 0:
                    if current_speaker != past_speaker:
                        line_features.append('ch')
                tokens_tags = val.pos

                if tokens_tags is not None:
                    line_features.append('beginWithPos=' + tokens_tags[0].pos)
                    line_features.append('beginWithToken=' + tokens_tags[0].token)

                if tokens_tags is not None:
                    for x in tokens_tags:
                        all_token.append(x.token)

                        line_features.append('TOKEN_' + x.token)
                    for x in tokens_tags:
                        all_pos.append(x.pos)
                        line_features.append('POS_' +x.pos)
                    bigram_token = self.get_bigram(all_token)
                    bigram_pos = self.get_bigram(all_pos)
                    trigram_token = self.get_trigram(all_token)
                    trigram_pos = self.get_trigram(all_pos)
                    for word1, word2 in bigram_token:
                        word = word1 + ',' + word2
                        line_features.append('BIT_' + word)
                    for word1, word2 in bigram_pos:
                        word = word1 + ',' + word2
                        line_features.append('BIP_' + word)
                    for word1, word2, word3 in trigram_token:
                        word = word1 + ',' + word2 + ',' + word3
                        line_features.append('TRT_' + word)
                    for word1, word2, word3 in trigram_pos:
                        word = word1 + ',' + word2 + ',' + word3
                        line_features.append('TGP_' + word)

                else:
                    line_features.append('NO_WORDS')

                if tokens_tags is not None:
                    line_features.append('endWithPos=' + tokens_tags[-1].pos)
                    line_features.append('endWithToken=' + tokens_tags[-1].token)
                utterance_label.append(val.act_tag)
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
                all_token = []
                all_pos = []
                line_features = []
                current_speaker = val.speaker
                if count == 0:
                    line_features.append('fs')
                if count > 0:
                    if current_speaker != past_speaker:
                        line_features.append('ch')
                tokens_tags = val.pos
                if tokens_tags is not None:
                    line_features.append('beginWithPos=' + tokens_tags[0].pos)
                    line_features.append('beginWithToken=' + tokens_tags[0].token)
                if tokens_tags is not None:
                    for x in tokens_tags:
                        line_features.append('TOKEN_' + x.token)
                        all_token.append(x.token)
                    for x in tokens_tags:
                        line_features.append('POS_' + x.pos)
                        all_pos.append(x.pos)
                    trigram_token = self.get_trigram(all_token)
                    trigram_pos = self.get_trigram(all_pos)
                    bigram_token = self.get_bigram(all_token)
                    bigram_pos = self.get_bigram(all_pos)
                    for word1, word2 in bigram_token:
                        word = word1 + ',' + word2
                        line_features.append('BIT_' + word)
                    for word1, word2 in bigram_pos:
                        word = word1 + ',' + word2
                        line_features.append('BIP_' + word)
                    for word1, word2, word3 in trigram_token:
                        word = word1 + ',' + word2 + ',' + word3
                        line_features.append('TRT_' + word)
                    for word1, word2, word3 in trigram_pos:
                        word = word1 + ',' + word2 + ',' + word3
                        line_features.append('TGP_' + word)

                else:
                    line_features.append('NO_WORDS')
                if tokens_tags is not None:
                    line_features.append('endWithPos=' + tokens_tags[-1].pos)
                    line_features.append('endWithToken=' + tokens_tags[-1].token)

                utterance_label.append(val.act_tag)
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
        print("Advanced Acccuracy = " + str(cor * 100 / sum) + '%\n')


if __name__ == '__main__':
    start_time = time.time()
    tag = Tagger()
    tag.train()
    print(' Input Features Loaded')
    tag.build_model()
    print('CRF Model trained on input data')
    tag.test()
    print('CRF Model tested on testing data')
    tag.check_accuracy()
    end_time = time.time()
    print('Time Taken', end_time - start_time)
