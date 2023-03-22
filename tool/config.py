class DefaultConfig (object) :
    use_gpu = True
    max_epoch =600
    learning_rate = 1e-3
    decay_LR = (550, 0.5)
    weight_decay = 5e-5
opt  = DefaultConfig()