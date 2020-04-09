import hw2_corpus_tool
import pycrfsuite
import time
import sys


# TODO ADD SUPPORT FOR ACT TAGS IN TEST DATA.
class Tagger:
    def __init__(self):
        # ram_train_path='/Users/nikhildhara/Desktop/nlp-tagger/ram_data/train'
        # ram_test_path='/Users/nikhildhara/Desktop/nlp-tagger/ram_data/dev'

        # self.train_set_path = '/Users/nikhildhara/Desktop/nlp-tagger/nik_test'
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
            index = 0
            utterance_label = []
            past_speaker = ''
            feature_list = []
            for sentence in utterance:
                all_token = []
                all_pos = []
                line_features = []
                current_speaker = sentence.speaker
                if index == 0:
                    line_features.append('fs')
                if index > 0:
                    if current_speaker != past_speaker:
                        line_features.append('ch')
                tokens_tags = sentence.pos

                if tokens_tags is not None:
                    line_features.append('startpos=' + tokens_tags[0].pos)
                    line_features.append('starttoken=' + tokens_tags[0].token)

                if tokens_tags is not None:
                    for x in tokens_tags:
                        all_token.append(x.token)

                        line_features.append('TOKEN_' + x.token)
                    for x in tokens_tags:
                        all_pos.append(x.pos)
                        line_features.append('POS_' + x.pos)
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
                    line_features.append('endpos=' + tokens_tags[-1].pos)
                    line_features.append('endtoken=' + tokens_tags[-1].token)
                utterance_label.append(sentence.act_tag)
                feature_list.append(line_features)
                index += 1
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

    def test(self):
        self.testing_data = list(hw2_corpus_tool.get_data(self.test_set_path))
        for utterance in self.testing_data:

            index = 0
            utterance_label = []
            past_speaker = ''
            feature_list = []
            for sentence in utterance:
                all_token = []
                all_pos = []
                line_features = []
                current_speaker = sentence.speaker
                if index == 0:
                    line_features.append('fs')
                if index > 0:
                    if current_speaker != past_speaker:
                        line_features.append('ch')
                tokens_tags = sentence.pos
                # if tokens_tags is not None:

                if tokens_tags is not None:
                    line_features.append('startpos=' + tokens_tags[0].pos)
                    line_features.append('starttoken=' + tokens_tags[0].token)
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
                    line_features.append('endpos=' + tokens_tags[-1].pos)
                    line_features.append('endtoken=' + tokens_tags[-1].token)

                else:
                    line_features.append('NO_WORDS')
                # if tokens_tags is not None:

                utterance_label.append(sentence.act_tag)
                feature_list.append(line_features)
                index += 1
                past_speaker = current_speaker
            self.testfeatures.append(feature_list)
            self.testlabels.append(utterance_label)

    # def check_accuracy(self):
    #     tagger = pycrfsuite.Tagger()
    #     tagger.open('tagger_model_trained')
    #     sum = 0
    #     cor = 0
    #     for i in range(len(self.testfeatures)):
    #         pred = tagger.tag(self.testfeatures[i])
    #         for j in range(len(pred)):
    #             if pred[j] == self.testlabels[i][j]:
    #                 cor += 1
    #             sum += 1
    #     print("Advanced Acccuracy = " + str(cor * 100 / sum) + '%\n')

    def write_output(self):
        tagger = pycrfsuite.Tagger()
        tagger.open('tagger_model_trained')
        output_file = open(self.output_file, 'w')
        for i in range(len(self.testfeatures)):
            pred = tagger.tag(self.testfeatures[i])
            for j in range(len(pred)):
                output_file.write(pred[j] + '\n')
            output_file.write('\n')
        output_file.close()


if __name__ == '__main__':
    start_time = time.time()
    tag = Tagger()
    tag.train()
    print(' Input Features Loaded')
    tag.build_model()
    print('CRF Model trained on input data')
    tag.test()
    print('CRF Model tested on testing data')
    # tag.check_accuracy()
    tag.write_output()
    end_time = time.time()
    print('Time Taken', end_time - start_time)
