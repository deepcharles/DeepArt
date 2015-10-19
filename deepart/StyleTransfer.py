__author__ = 'charles'
from build_model import build_model
from image_utils import prep_image, deprocess
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
import lasagne
from deepart import floatX, np, IMAGE_W
from loss_functions import content_loss, style_loss, total_variation_loss
import scipy


class StyleTransfer:
    def __init__(self, photo_string, art_string, content=0.001, style=0.2e6, total_var=0.1e-7):
        # load network
        self.net = build_model()
        # load layers
        layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        layers = {k: self.net[k] for k in layers}
        self.layers = layers
        # load images
        im = plt.imread(art_string)
        self.art_raw, self.art = prep_image(im)
        im = plt.imread(photo_string)
        self.photo_raw, self.photo = prep_image(im)
        # precompute layer activations for photo and artwork
        input_im_theano = T.tensor4()
        self._outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
        self.photo_features = {k: theano.shared(output.eval({input_im_theano: self.photo}))
                          for k, output in zip(layers.keys(), self._outputs)}
        self.art_features = {k: theano.shared(output.eval({input_im_theano: self.art}))
                        for k, output in zip(layers.keys(), self._outputs)}
        # Get expressions for layer activations for generated image
        self.generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

        gen_features = lasagne.layers.get_output(layers.values(), self.generated_image)
        gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
        self.gen_features = gen_features

        # set the weights of the regularizers
        self._content, self._style, self._total_var = content, style, total_var


    @property
    def total_loss(self):
        res = []
        # content loss
        res.append(self._content * content_loss(self.photo_features, self.gen_features, 'conv4_2'))

        # style loss
        res.append(self._style * style_loss(self.art_features, self.gen_features, 'conv1_1'))
        res.append(self._style * style_loss(self.art_features, self.gen_features, 'conv2_1'))
        res.append(self._style * style_loss(self.art_features, self.gen_features, 'conv3_1'))
        res.append(self._style * style_loss(self.art_features, self.gen_features, 'conv4_1'))
        res.append(self._style * style_loss(self.art_features, self.gen_features, 'conv5_1'))

        # total variation penalty
        res.append(self._total_var * total_variation_loss(self.generated_image))

        return sum(res)

    @property
    def grad(self):
        return T.grad(self.total_loss, self.generated_image)


    def eval_loss(self, x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        self.generated_image.set_value(x0)
        f_loss = theano.function([], self.total_loss)
        return f_loss().astype('float64')


    def eval_grad(self, x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        self.generated_image.set_value(x0)
        f_grad = theano.function([], self.grad)
        return np.array(f_grad()).flatten().astype('float64')

    def transfer_style(self, init=None, saveplot=True):
        if init is None:
            self.generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))
        else:
            im = plt.imread(init)
            _, tmp_im = prep_image(im)
            self.generated_image.set_value(tmp_im)

        x0 = self.generated_image.get_value().astype('float64')

        # Optimize
        print()
        print("Starting optimization.")
        scipy.optimize.fmin_l_bfgs_b(self.eval_loss, x0.flatten(), fprime=self.eval_grad, maxfun=400)
        print()
        print("Done")
        x0 = self.generated_image.get_value().astype('float64')
        im = deprocess(x0)

        if saveplot:
            im = self.transfer_style(init=init)
            plt.gca().xaxis.set_visible(False)
            plt.gca().yaxis.set_visible(False)
            plt.imshow(im)
            plt.savefig("style.jpg")
        return im

if __name__ == '__main__':
    import time
    tic = time.time()
    s = StyleTransfer(photo_string="photo_charles.jpg",
                      art_string="1920px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg")
    print("Instantiation in " + str(time.time() - tic))
    print()
    tic = time.time()
    s.transfer_style()
    print()
    print("Style transfer done in: " + str(time.time() - tic))
