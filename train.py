import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
sys.path.append(r'D:\adversarial_machine_translation')
from model.G import Encoder, Decoder, Seq2Seq
from model.D import Discriminator
from utilizer.utilizer import load_dataset, max_length
from sklearn.model_selection import train_test_split
import numpy as np
import time

def gradient_penalty(discriminator, de_onehot, en_onehot, gen_en):
    batchsz = en_onehot.shape[0]
    t = tf.random.uniform([batchsz, 1, 1])
    t = tf.broadcast_to(t, en_onehot.shape)
    interplate = t * en_onehot + (1 - t) * gen_en
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator([de_onehot, interplate])
    grads = tape.gradient(d_interplote_logits, interplate)
    grads = tf.reshape(grads, [grads.shape[0], -1]) # 梯度矩阵展平
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    sentence_loss = tf.reduce_sum(loss_, axis=1)
    return tf.reduce_mean(sentence_loss)


def d_loss_fn(generator, discriminator, de, en):
    # 1. treat [de - en] pair as real
    # 2. treat [de - gen_en] pair as fake(gen)

    de_onehot = tf.one_hot(de, depth=vocab_de_size)
    en_onehot = tf.one_hot(en, depth=vocab_en_size)

    # 1. real
    d_real_logits = discriminator([de_onehot, en_onehot])

    # 2. gen
    gen_en = generator(de, en)
    d_gen_logits = discriminator([de_onehot, gen_en])

    # total loss
    gp = gradient_penalty(discriminator, de_onehot, en_onehot, gen_en)
    loss = - tf.reduce_mean(d_real_logits) + tf.reduce_mean(d_gen_logits) + 10. * gp
    return loss, gp


def g_loss_fn(generator, discriminator, de, en):

    de_onehot = tf.one_hot(de, depth=vocab_de_size)

    # translation loss
    gen_en = generator(de, en)
    loss1 = loss_function(en,gen_en)

    # wgan-gp loss
    d_gen_logits = discriminator([de_onehot, gen_en])
    loss2 = - tf.reduce_mean(d_gen_logits)

    # total loss
    loss = (10 * loss1) + loss2
    return loss


if __name__ == '__main__':


    # ------------------------------------------------------------------------------
    file_path = "Corpus/ind.txt"

    num_sentence = 7000
    filename = 'ind'
    test_size = 0.2

    print(filename)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


    tf.random.set_seed(13)
    np.random.seed(13)
    BATCH_SIZE = 512
    # read data
    de_tensor, en_tensor, de_dic, en_dic = load_dataset(file_path, num_examples=num_sentence)
    de_tensor_train, de_tensor_val, en_tensor_train, en_tensor_val = train_test_split(de_tensor,
                                                                                      en_tensor,
                                                                                      test_size=test_size)
    Buffer_Size = len(de_tensor_train)
    train_db = tf.data.Dataset.from_tensor_slices((de_tensor_train, en_tensor_train)).shuffle(Buffer_Size)
    train_db = train_db.batch(BATCH_SIZE, drop_remainder=True)

    val_db = tf.data.Dataset.from_tensor_slices((de_tensor_val, en_tensor_val)).shuffle(Buffer_Size)
    val_db = val_db.batch(BATCH_SIZE, drop_remainder=True)

    max_length_en, max_length_de = max_length(en_tensor), max_length(de_tensor)

    # hyper parameters
    embedding_dim = 128
    cov_dim = 128
    units = 1024
    vocab_de_size = 4400
    vocab_en_size = 3400
    epochs = 801


    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    opt = tf.keras.optimizers.Adam()
    # ------------------------------------------------------------------------------

    # same embedding_dim in DE and EN
    encoder = Encoder(vocab_de_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_en_size, embedding_dim, units, BATCH_SIZE)
    generator = Seq2Seq(encoder, decoder, vocab_en_size)
    # generator.load_weights('D:\\PycharmProjects\\paper2 v10\\1_BahdanauRNN\\土耳其\weight\\generator')
    discriminator = Discriminator(vocab_de_len= max_length_de,
                                  vocab_en_len= max_length_en,
                                  vocab_de_size= vocab_de_size,
                                  vocab_en_size= vocab_en_size,
                                  embedding_dim = embedding_dim,
                                  cov_dim = cov_dim)

    for epoch in range(epochs):
        for b, (de_batch, en_batch) in enumerate(iter(train_db)):

        # train D
            with tf.GradientTape() as tape:
                d_loss, gp = d_loss_fn(generator, discriminator, de_batch, en_batch)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            opt.apply_gradients(zip(grads, discriminator.trainable_variables))

            # weight clipping
            # for l in discriminator.layers:
            #     weights = l.get_weights()
            #     weights = [tf.clip_by_value(w, -0.01, 0.01) for w in weights]
            #     l.set_weights(weights)

        # train G
            if b % 10 == 0:
                with tf.GradientTape() as tape:
                    g_loss = g_loss_fn(generator, discriminator, de_batch, en_batch)
                grads = tape.gradient(g_loss, generator.trainable_variables)
                opt.apply_gradients(zip(grads, generator.trainable_variables))


        if epoch % 1 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss),
                  'w-distance:', float((d_loss) - (g_loss)))

        # eval
        if epoch % 10 == 0:
            for b, (example_input_batch, example_target_batch) in enumerate(iter(val_db)):
                trans_en = generator(example_input_batch, example_target_batch, teacher_forcing_ratio=0)
                for eb in range(BATCH_SIZE):
                    translation = ''
                    origin = ''

                    predicted_id = (tf.argmax(trans_en, -1)[eb].numpy())
                    tar_id = example_target_batch[eb].numpy()

                    for t in range(max_length_en):
                        # we unify the vocabulary scale
                        # so it may generate non-existent word in the beginning.
                        try:
                            if predicted_id[t] == 0:
                                break
                            if en_dic.index_word[predicted_id[t]] == '<end>':
                                break
                            if en_dic.index_word[predicted_id[t]] == '<start>':
                                continue
                            translation += en_dic.index_word[predicted_id[t]] + ' '
                        except:
                            break

                    for t in range(max_length_en):
                        try:
                            if en_dic.index_word[tar_id[t]] == '<end>':
                                break
                            if en_dic.index_word[tar_id[t]] == '<start>':
                                continue
                            origin += en_dic.index_word[tar_id[t]] + ' '
                        except:
                            break
                    with open('model/%s/result/epoch %d.log' % (filename,epoch), 'a') as f:
                        f.writelines("%s\n%s\n\n"
                                     % (origin, translation))
                    print(origin)
                    print(translation)
            print("iter %d times" % epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


        # save generator and discriminator at each 20 interval
        if epoch % 20 == 0:
            generator.save_weights('model/%s\G_weight\generator'% filename, overwrite=True)
            discriminator.save_weights('model/%s\D_weight\discriminator'% filename, overwrite=True)