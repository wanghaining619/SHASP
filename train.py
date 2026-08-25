"""Train SHASP on VIS/IR image collections."""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size, flush=True)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        adversarial_scale = (
            model._adversarial_scale()
            if hasattr(model, '_adversarial_scale') else 1.0
        )
        if adversarial_scale == 0:
            training_phase = 'paired warmup'
        elif adversarial_scale < 1:
            training_phase = 'adversarial ramp'
        else:
            training_phase = 'full objective'
        current_lr = model.optimizers[0].param_groups[0]['lr']
        print(
            '\n===== Epoch {}/{} | phase: {} | adversarial scale: {:.2f} '
            '| lr: {:.7f} ====='.format(
                epoch, opt.n_epochs + opt.n_epochs_decay, training_phase,
                adversarial_scale, current_lr
            ),
            flush=True
        )
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print(
                    'saving the latest model (epoch %d, total_iters %d)'
                    % (epoch, total_iters),
                    flush=True
                )
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print(
                'saving the model at the end of epoch %d, iters %d'
                % (epoch, total_iters),
                flush=True
            )
            model.save_networks('latest')
            model.save_networks(epoch)

        # Step after the optimizer has run for the epoch. This preserves the
        # full initial learning-rate interval on modern PyTorch versions.
        model.update_learning_rate()
        print(
            'End of epoch %d / %d \t Time Taken: %d sec'
            % (
                epoch, opt.n_epochs + opt.n_epochs_decay,
                time.time() - epoch_start_time
            ),
            flush=True
        )
