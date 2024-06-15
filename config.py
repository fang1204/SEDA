import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.train_file = config["train_file"]
        self.dev_file = config["dev_file"]
        self.test_file = config["test_file"]
        self.save_path = config["save_path"]
        self.enhance_project = config["enhance_project"]
        self.generate_mode = config["generate_mode"]
        self.enhance_count = config["enhance_count"]

        self.ES_action_state = config["ES_action_state"]
        self.NES_action_state = config["NES_action_state"]
        self.contain_merge_conflict_span = config["contain_merge_conflict_span"]
        self.support_strategy = config["support_strategy"]
        self.look_forward_step = config["look_forward_step"]
        self.look_backward_step = config["look_backward_step"]

        # self.previous_path = config["previous_path"]
        self.predict_path = config["predict_path"]
        self.accumulate_step = config["accumulate_step"]
        self.entity_compare = config["entity_compare"]
        self.dist_emb_size = config["dist_emb_size"]
        self.type_emb_size = config["type_emb_size"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.conv_hid_size = config["conv_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.biaffine_size = config["biaffine_size"]
        self.ffnn_hid_size = config["ffnn_hid_size"]

        self.dilation = config["dilation"]

        self.emb_dropout = config["emb_dropout"]
        self.conv_dropout = config["conv_dropout"]
        self.out_dropout = config["out_dropout"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.use_bert_last_4_layers = config["use_bert_last_4_layers"]

        self.seed = config["seed"]

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
