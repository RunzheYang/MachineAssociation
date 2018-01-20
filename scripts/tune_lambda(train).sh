# delta = fearture_l1, eta = nll
python refine_apprentice.py --refiner unet --lbd 0.00 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.00
python refine_apprentice.py --refiner unet --lbd 0.05 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.05
python refine_apprentice.py --refiner unet --lbd 0.10 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.10
python refine_apprentice.py --refiner unet --lbd 0.15 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.15
python refine_apprentice.py --refiner unet --lbd 0.20 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.20
python refine_apprentice.py --refiner unet --lbd 0.25 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.25
python refine_apprentice.py --refiner unet --lbd 0.30 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.30
python refine_apprentice.py --refiner unet --lbd 0.35 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.35
python refine_apprentice.py --refiner unet --lbd 0.40 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.40
python refine_apprentice.py --refiner unet --lbd 0.45 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.45
python refine_apprentice.py --refiner unet --lbd 0.50 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.50
python refine_apprentice.py --refiner unet --lbd 0.55 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.55
python refine_apprentice.py --refiner unet --lbd 0.60 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.60
python refine_apprentice.py --refiner unet --lbd 0.65 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.65
python refine_apprentice.py --refiner unet --lbd 0.70 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.70
python refine_apprentice.py --refiner unet --lbd 0.75 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.75
python refine_apprentice.py --refiner unet --lbd 0.85 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.85
python refine_apprentice.py --refiner unet --lbd 0.90 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.90
python refine_apprentice.py --refiner unet --lbd 0.95 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd0.95
python refine_apprentice.py --refiner unet --lbd 1.00 --delta feature_l1 --eta nll --epochs 500 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet_lmd1.00