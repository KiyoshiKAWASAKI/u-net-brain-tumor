#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time
import gan_model as model

def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y]), size=(1, 5),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)

def main(task='all'):
    ## Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    import prepare_data_with_valid as dataset
    X_train = dataset.X_train_input
    y_train = dataset.X_train_target[:,:,:,np.newaxis]
    X_test = dataset.X_dev_input
    y_test = dataset.X_dev_target[:,:,:,np.newaxis]


    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
    else:
        exit("Unknow task %s" % task)

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 4
    lr = 0.0001 
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 50
    print_freq_step = 1
    kt = tf.Variable(0., trainable=False)
    gamma = 0.75
    lamda = 0.001

    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[80])
    y = np.asarray(y_train[80])
    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    nw, nh, nz = X.shape
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:,:,0,np.newaxis], X[:,:,1,np.newaxis],
                X[:,:,2,np.newaxis], X[:,:,3,np.newaxis], y])#[:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))

    with tf.device('/gpu:2'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:2'): #<- remove it if you train on CPU or other GPU
            ###======================== DEFIINE MODEL =======================###
            #tf.reset_default_graph()
            ## nz is 4 as we input all Flair, T1, T1c and T2.
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
            
            ## Training (using generator)
            net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
            net_result = net.outputs #[batch, 240, 240, 1]
            
            # Concat the  generated results with ground truth label
            #concated = tf.concat([t_seg, net_result], axis=3) # [10, 240, 240, 2]
            concated = tf.concat([t_image, net_result], axis=3) # [10, 240, 240, 5]
            #real_target = tf.concat([t_seg, t_seg], axis=3)
            real_target = tf.concat([t_image, t_seg], axis=3) # [10, 240, 240, 5]
            
            # Send concated tensor into discriminator
            d_out = model.discriminator(concated, is_train=True, reuse = False)
            d_real = model.discriminator(real_target, is_train=True, reuse = True)
            
            ## test inference
            net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)

            ###======================== DEFINE LOSS =========================###
            ## Train losses
            recons_loss = 1 - tl.cost.dice_coe(d_out, concated, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            fake_iou_loss = tl.cost.iou_coe(d_out, concated, axis=[0,1,2,3])
            fake_dice_hard = tl.cost.dice_hard_coe(d_out, concated, axis=[0,1,2,3])

            real_dice_loss = 1 - tl.cost.dice_coe(d_real, real_target, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            real_iou_loss = tl.cost.iou_coe(d_real, real_target, axis=[0,1,2,3])
            real_dice_hard = tl.cost.dice_hard_coe(d_real, real_target, axis=[0,1,2,3])

            # Summary of G and D losses
            G_loss = recons_loss
            D_loss = real_dice_loss - kt * recons_loss
            M = real_dice_loss+tf.abs(gamma * real_dice_loss - recons_loss)
            k_update = kt.assign(kt + lamda * (gamma * real_dice_loss - recons_loss))

            ## test losses
            test_out_seg = net_test.outputs
            test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
            test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])

        ###======================== DEFINE TRAIN OPTS =======================###
        g_vars = tl.layers.get_variables_with_name('u_net', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            g_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(G_loss, var_list=g_vars)

        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            d_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(D_loss, var_list=d_vars)

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

        ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        n_batch = 0
        total_dice_fake, total_iou_fake, total_dice_hard_fake = 0, 0, 0
        total_dice_real, total_iou_real, total_dice_hard_real = 0, 0, 0
        
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                    images[:,:,:,3, np.newaxis], labels)],
                    fn=distort_imgs) # (10, 5, 240, 240, 1)
            b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,4,:,:,:]
            b_images = b_images.transpose((0,2,3,1,4))
            b_images.shape = (batch_size, nw, nh, nz)

            ## update network
            # Run generater
            _, _fakedice, _fakeiou, \
            _fakediceh, out = sess.run([g_op, recons_loss, fake_iou_loss,
                                    fake_dice_hard,net.outputs],
                                    {t_image: b_images, t_seg: b_labels})
            total_dice_fake += _fakedice
            total_iou_fake += _fakeiou
            total_dice_hard_fake += _fakediceh
            
            # Run discriminator
            _, _realdice, \
            _realiou, _realdiceh = sess.run([d_op, real_dice_loss,
                                real_iou_loss, real_dice_hard],
                                {t_image: b_images, t_seg: b_labels})
            total_dice_real += _realdice
            total_iou_real += _realiou
            total_dice_hard_real += _realdiceh


            # update k and losses
            _, convergence_metric, kt_for_print = sess.run([k_update, M, kt], 
                                                feed_dict={t_image: b_images, t_seg: b_labels})

            _, loss_G = sess.run([g_op, G_loss],
                                {t_image: b_images, t_seg: b_labels})

            _, loss_D = sess.run([g_op, D_loss],
                                {t_image: b_images, t_seg: b_labels})

            n_batch += 1

            ## you can show the predition here:
            # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            # exit()

            # if _dice == 1: # DEBUG
            #     print("DEBUG")
            #     vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

            if n_batch % print_freq_step == 0:

                print("Epoch %d step %d. G loss: %f; D loss: %f; M is%f; kt is %f"
                % (epoch, n_batch, loss_G, loss_D, convergence_metric, kt_for_print))
                print("Fake dice-hard: %f; Fake IOU: %f"
                % (_fakediceh, _fakeiou))
                print("Real dice-hard: %f; Real IOU: %f"
                % (_realdiceh, _realiou))

            ## check model fail
            if np.isnan(loss_G):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(loss_D):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        #print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                #(epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))


        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))

        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        print("Evaluation:"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        print(" task: {}".format(task))
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))

        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}.npz'.format(task), sess=sess)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')

    args = parser.parse_args()

    main(args.task)
