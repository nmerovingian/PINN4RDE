import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network):


        self.network = network

        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)

    def build(self):
        TYR_dmn0 = tf.keras.layers.Input(shape=(3,))
        aux1_dmn0 = tf.keras.layers.Input(shape=(1,))
        aux2_dmn0 = tf.keras.layers.Input(shape=(1,))
        aux3_dmn0 = tf.keras.layers.Input(shape=(1,))

        TYR_bnd0 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd1 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd2 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd3 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd4 = tf.keras.layers.Input(shape=(3,))

        Cdmn0, dC_dT_dmn0, dC_dY_dmn0,dC_dR_dmn0, d2C_dY2_dmn0,d2C_dR2_dmn0 = self.grads(TYR_dmn0)
        C_dmn0 = d2C_dY2_dmn0 + d2C_dR2_dmn0 + aux1_dmn0*dC_dR_dmn0 + aux2_dmn0*dC_dY_dmn0 - aux3_dmn0 * dC_dR_dmn0

        C_bnd0 = self.network(TYR_bnd0)

        Cbnd1,dC_dT_bnd1,dC_dY_bnd1,dC_dR_bnd1 = self.boundaryGrad(TYR_bnd1)
        C_bnd1 = dC_dY_bnd1

        C_bnd2 = self.network(TYR_bnd2)
        C_bnd3 = self.network(TYR_bnd3)
        C_bnd4 = self.network(TYR_bnd4)


        return tf.keras.models.Model(
            inputs=[TYR_dmn0,aux1_dmn0,aux2_dmn0,aux3_dmn0,TYR_bnd0,TYR_bnd1,TYR_bnd2,TYR_bnd3,TYR_bnd4], outputs=[C_dmn0,C_bnd0,C_bnd1,C_bnd2,C_bnd3,C_bnd4])
    


    def build_no_flux(self):


        TYR_dmn0 = tf.keras.layers.Input(shape=(3,))
        aux1_dmn0 = tf.keras.layers.Input(shape=(1,))
        aux2_dmn0 = tf.keras.layers.Input(shape=(1,))
        aux3_dmn0 = tf.keras.layers.Input(shape=(1,))

        TYR_bnd0 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd1 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd2 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd3 = tf.keras.layers.Input(shape=(3,))
        TYR_bnd4 = tf.keras.layers.Input(shape=(3,))


        Cdmn0, dC_dT_dmn0, dC_dY_dmn0,dC_dR_dmn0, d2C_dY2_dmn0,d2C_dR2_dmn0 = self.grads(TYR_dmn0)
        C_dmn0 = d2C_dY2_dmn0 + d2C_dR2_dmn0 + aux1_dmn0*dC_dR_dmn0 + aux2_dmn0*dC_dY_dmn0 - aux3_dmn0 * dC_dR_dmn0

        C_bnd0 = self.network(TYR_bnd0)

        Cbnd1,dC_dT_bnd1,dC_dY_bnd1,dC_dR_bnd1 = self.boundaryGrad(TYR_bnd1)
        C_bnd1 = dC_dY_bnd1

        C_bnd2 = self.network(TYR_bnd2)
        C_bnd3 = self.network(TYR_bnd3)
        Cbnd4,dC_dT_bnd4,dC_dY_bnd4,dC_dR_bnd4 = self.boundaryGrad(TYR_bnd4)
        C_bnd4 = dC_dR_bnd4







        return tf.keras.models.Model(
            inputs=[TYR_dmn0,aux1_dmn0,aux2_dmn0,aux3_dmn0,TYR_bnd0,TYR_bnd1,TYR_bnd2,TYR_bnd3,TYR_bnd4], outputs=[C_dmn0,C_bnd0,C_bnd1,C_bnd2,C_bnd3,C_bnd4])