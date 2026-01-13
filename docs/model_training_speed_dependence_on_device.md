At commit (1cd8f3d1f81c), the following comparison was done between the RL methods to investigate which models should be trained on gpu vs cpu.
shim2018:
    afa_context:
        cpu: 2-3 it/s
        gpu: 1 it/s
    synthetic_mnist:
        cpu: 4 it/s
        gpu: 1 it/s
kachuee2019:
    afa_context:
        cpu: 3 s/it
        gpu: 5 it/s
    synthetic_mnist:
        cpu: 40 s/it (!)
        gpu: OOM (1-2 it/s with RB device = cpu)
zannone2019:
    afa_context:
        cpu: 3 s/it
        gpu: 7-13 it/s
    synthetic_mnist:
        cpu: 3 s/it
        gpu: 3 it/s

Conclusion: both kachuee2019 and zannone2019 benefit from gpu training, but not shim2018.
