import tensorflow as tf
from .layer import GradientLayer

class PINN:


    def __init__(self, network):

        self.network = network
        self.grads = GradientLayer(self.network)

    def build(self):


        # equation input: (t, x)
        TY_eqn = tf.keras.layers.Input(shape=(2,))
        prefactor_eqn = tf.keras.layers.Input(shape=(1,))

        TY_bnd0 = tf.keras.layers.Input(shape=(2,))
        TY_bnd1 = tf.keras.layers.Input(shape=(2,))

        TY_ini = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        C, dC_dT, dC_dY, d2C_dY2 = self.grads(TY_eqn)

        # equation output being zero
        C_eqn = dC_dT - prefactor_eqn*d2C_dY2

        Cbnd0,dC_dT_bnd0,dC_dY_bnd0, d2C_dY2_bnd0 = self.grads(TY_bnd0)
        C_bnd0 = Cbnd0
        Cbnd1,dC_dT_bnd1,dC_dY_bnd1,d2C_dY2_bnd1 = self.grads(TY_bnd1)
        C_bnd1 = Cbnd1
        Cini,dC_dT_ini,dC_dY_ini,d2C_dY2_ini = self.grads(TY_ini)
        C_ini = Cini




        





        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[TY_eqn,prefactor_eqn,TY_bnd0,TY_bnd1,TY_ini], outputs=[C_eqn,C_bnd0,C_bnd1,C_ini])
