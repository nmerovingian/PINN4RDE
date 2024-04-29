import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network):


        self.network = network

        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)

    def build(self,NO_FLUX_BND,RADIALDIFFUSION):
        TYR_dmn0 = tf.keras.layers.Input(shape=(3,))
        TYR_dmn1 = tf.keras.layers.Input(shape=(3,))

        aux1overR_dmn0 = tf.keras.layers.Input(shape=(1,))
        auxY_dmn0 = tf.keras.layers.Input(shape=(1,))
        auxR_dmn0 = tf.keras.layers.Input(shape=(1,))

        aux1overR_dmn1 = tf.keras.layers.Input(shape=(1,))
        auxY_dmn1 = tf.keras.layers.Input(shape=(1,))
        auxR_dmn1 = tf.keras.layers.Input(shape=(1,))

        TYR_bnd0 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd1 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd2 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd3 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd4 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd5 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd6 = tf.keras.layers.Input(shape=(3,))


        Cdmn0, dC_dT_dmn0, dC_dY_dmn0,dC_dR_dmn0, d2C_dY2_dmn0,d2C_dR2_dmn0 = self.grads(TYR_dmn0)
        Cdmn1, dC_dT_dmn1, dC_dY_dmn1,dC_dR_dmn1, d2C_dY2_dmn1,d2C_dR2_dmn1 = self.grads(TYR_dmn1)

        if RADIALDIFFUSION:
            C_dmn0 = d2C_dY2_dmn0 + d2C_dR2_dmn0 + aux1overR_dmn0*dC_dR_dmn0 - auxY_dmn0*dC_dY_dmn0 - auxR_dmn0 * dC_dR_dmn0
            C_dmn1 = d2C_dY2_dmn1 + d2C_dR2_dmn1 + aux1overR_dmn1*dC_dR_dmn1 - auxY_dmn1*dC_dY_dmn1 - auxR_dmn1 * dC_dR_dmn1
        else:
            C_dmn0 = d2C_dY2_dmn0 - auxY_dmn0*dC_dY_dmn0 - auxR_dmn0 * dC_dR_dmn0
            C_dmn1 = d2C_dY2_dmn1 - auxY_dmn1*dC_dY_dmn1 - auxR_dmn1 * dC_dR_dmn1

        C_dmn0  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_dmn0')(C_dmn0)
        C_dmn1  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_dmn1')(C_dmn1)

        C_bnd0 = self.network(TYR_bnd0)
        C_bnd0  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd0')(C_bnd0)


        Cbnd1,dC_dT_bnd1,dC_dY_bnd1,dC_dR_bnd1 = self.boundaryGrad(TYR_bnd1)
        C_bnd1 = dC_dY_bnd1
        C_bnd1  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd1')(C_bnd1)


        C_bnd2 = self.network(TYR_bnd2)
        C_bnd2  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd2')(C_bnd2)


        Cbnd3,dC_dT_bnd3,dC_dY_bnd3,dC_dR_bnd3 = self.boundaryGrad(TYR_bnd3)
        C_bnd3 = dC_dY_bnd3
        C_bnd3  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd3')(C_bnd3)


        Cbnd4,dC_dT_bnd4,dC_dY_bnd4,dC_dR_bnd4 = self.boundaryGrad(TYR_bnd4)
        if NO_FLUX_BND:
            C_bnd4 = dC_dR_bnd4
        else:
            C_bnd4 = Cbnd4
        C_bnd4  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd4')(C_bnd4)


        C_bnd5 = self.network(TYR_bnd5)
        C_bnd5  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd5')(C_bnd5)


        C_bnd6 = self.network(TYR_bnd6)
        C_bnd6  = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='C_bnd6')(C_bnd6)


        return tf.keras.models.Model(
            inputs=[TYR_dmn0,aux1overR_dmn0,auxY_dmn0,auxR_dmn0,TYR_dmn1,aux1overR_dmn1,auxY_dmn1,auxR_dmn1,TYR_bnd0,TYR_bnd1,TYR_bnd2,TYR_bnd3,TYR_bnd4,TYR_bnd5,TYR_bnd6], outputs=[C_dmn0,C_dmn1,C_bnd0,C_bnd1,C_bnd2,C_bnd3,C_bnd4,C_bnd5,C_bnd6])
    


