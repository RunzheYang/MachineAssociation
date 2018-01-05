python refine_apprentice.py --refiner unet --lbd 0.95 --epochs 200 --optimizer Adam --lr 1e-4 --batch-size 50 --test-batch 100 --test-size 200 --classifier-path classifier/saved/ --classifier-name mnist_lenet --save refiner/saved/ --log refiner/logs/ --name mnist_unet