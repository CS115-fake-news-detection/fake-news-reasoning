from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
# from transformers import DistilBertTokenizerFast, DistilBertPreTrainedModel, DistilBertModel
from parameters import BERT_MODEL_PATH, CLAIM_ONLY, CLAIM_AND_EVIDENCE, EVIDENCE_ONLY, DEVICE
from torch.nn import functional as F
import torch.nn as nn
import torch
import string
import pandas as pd


class MyBertModel(BertPreTrainedModel):
    def __init__(self, config, labelnum, maxlen=200, input_type=CLAIM_ONLY):

        super(MyBertModel, self).__init__(config)

        self.emocred_type = 'EMO_INT'
        self.input_type = input_type
        self.maxlen = maxlen

        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH)

        self.bert = BertModel(config)

        if input_type == CLAIM_ONLY or input_type == EVIDENCE_ONLY:
            self.predictor = nn.Linear(768 + 8, labelnum)
        elif input_type == CLAIM_AND_EVIDENCE:
            self.predictor = nn.Linear(768 * 2 + 16, labelnum)

        if self.input_type != CLAIM_ONLY:
            self.attn_score = nn.Linear(768, 1)
            self.softmax = nn.Softmax(dim=1)

        self.emolinear1 = nn.Linear(8, 8)

        lex_path = "model/NRC-Emotion-Intensity-Lexicon-v1.txt"
        df_lex = pd.read_csv(lex_path, sep='\t', index_col=False)

        self.LEXICON = {
            k: v
            for (k, v) in zip(
                df_lex['word'],
                zip(df_lex['emotion'], df_lex['emotion-intensity-score']))
        }
        self.EMOTION_INDICES = {
            'anger': 0,
            'fear': 1,
            'joy': 2,
            'sadness': 3,
            'disgust': 4,
            'anticipation': 5,
            'trust': 6,
            'surprise': 7
        }

        self.init_weights()

    def forward(self, claims, snippets):
        if self.input_type == CLAIM_ONLY:
            return self.predict_claim(claims)
        elif self.input_type == EVIDENCE_ONLY:
            return self.predict_evidence(snippets)
        elif self.input_type == CLAIM_AND_EVIDENCE:
            return self.predict_claim_evidence(claims, snippets)
        else:
            raise Exception("Unknown type", self.input_type)

    def encode_claims(self, claims):
        tmp = self.tokenizer(claims,
                             return_tensors='pt',
                             padding=True,
                             truncation=True,
                             max_length=self.maxlen)
        input_ids = tmp["input_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)

        return input_ids, attention_mask

    def encode_snippets(self, snippets):
        concat_snippets = [
            item for sublist in snippets for item in sublist.tolist()
        ]
        tmp = self.tokenizer(concat_snippets,
                             return_tensors='pt',
                             padding=True,
                             truncation=True,
                             max_length=self.maxlen)
        input_ids = tmp["input_ids"].to(DEVICE)
        token_type_ids = tmp["token_type_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)
        return input_ids, token_type_ids, attention_mask

    def encode_snippets_with_claims(self, snippets, claims):
        concat_claims = []
        for claim in claims:
            concat_claims += [claim] * 10

        concat_snippets = [
            item for sublist in snippets for item in sublist.tolist()
        ]

        tmp = self.tokenizer(concat_claims,
                             concat_snippets,
                             return_tensors='pt',
                             padding=True,
                             truncation=True,
                             max_length=self.maxlen)

        input_ids = tmp["input_ids"].to(DEVICE)
        token_type_ids = tmp["token_type_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)

        return input_ids, token_type_ids, attention_mask

    def predict_claim(self, claims):
        claim_input_ids, claim_attn_masks = self.encode_claims(claims)
        emotion_vector = self.emocred(claims, self.emocred_type)
        emotion_vector = self.emolinear1(emotion_vector)
        cls = self.bert(
            claim_input_ids,
            attention_mask=claim_attn_masks,
        )[0][:, 0, :]
        return self.predictor(torch.cat((cls, emotion_vector), dim=1))

    def predict_evidence(self, snippets):

        emotion_vectors = torch.zeros(len(snippets), 10, 8).to(DEVICE)

        for i, snippet in enumerate(snippets):
            snippet = snippet.tolist()
            emotion_vector = self.emocred(snippet, self.emocred_type)
            emotion_vectors[i] = emotion_vector

        snippet_input_ids, snippet_token_type_ids, snippet_attention_mask = self.encode_snippets(
            snippets)
        snippet_cls = self.bert(snippet_input_ids,
                                token_type_ids=snippet_token_type_ids,
                                attention_mask=snippet_attention_mask)[0][:,
                                                                          0, :]
        snippet_cls = snippet_cls.view(len(snippets), 10, 768)

        tmp = self.attn_score(snippet_cls)
        attn_weights = self.softmax(tmp)
        snippet_cls = snippet_cls * attn_weights
        snippet_cls = torch.sum(snippet_cls, dim=1)
        emotion_vectors = torch.sum(emotion_vectors, dim=1)

        print(f"SNIPPET SHAPE: {snippet_cls.shape}")
        print(f"EMOTION SHAPE: {emotion_vectors.shape}")

        return self.predictor(torch.cat((snippet_cls, emotion_vectors),
                                        dim=-1))

    def predict_claim_evidence(self, claims, snippets):
        claim_input_ids, claim_attn_masks = self.encode_claims(claims)
        claim_cls = self.bert(claim_input_ids,
                              attention_mask=claim_attn_masks)[0][:, 0, :]

        snippet_input_ids, snippet_token_type_ids, snippet_attention_mask = self.encode_snippets_with_claims(
            snippets, claims)
        snippet_cls = self.bert(snippet_input_ids,
                                token_type_ids=snippet_token_type_ids,
                                attention_mask=snippet_attention_mask)[0][:,
                                                                          0, :]
        snippet_cls = snippet_cls.view(len(claims), 10, 768)

        tmp = self.attn_score(snippet_cls)
        attn_weights = self.softmax(tmp)
        snippet_cls *= attn_weights
        snippet_cls = torch.sum(snippet_cls, dim=1)

        claim_snippet_cls = torch.cat((claim_cls, snippet_cls), dim=-1)

        return self.predictor(claim_snippet_cls)

    def emocred(self, texts, emocred_type):
        emo_lexi = torch.zeros(len(texts), 8)
        emo_int = torch.zeros(len(texts), 8)
        for index, text in enumerate(texts):
            tokens = text.lower().translate(
                str.maketrans('', '', string.punctuation)).strip().split()
            for word in tokens:
                if word in self.LEXICON:
                    #print(word, self.LEXICON[word][0], self.LEXICON[word][1])
                    emo_lexi[index,
                             self.EMOTION_INDICES[self.LEXICON[word][0]]] += 1
                    emo_int[index, self.EMOTION_INDICES[
                        self.LEXICON[word][0]]] += self.LEXICON[word][1]

        if self.emocred_type == 'EMO_INT':
            return emo_int.to(DEVICE)
        elif self.emocred_type == 'EMO_LEXI':
            return emo_lexi.to(DEVICE)
