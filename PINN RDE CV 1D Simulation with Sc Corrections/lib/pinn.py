import tensorflow as tf
from .layer import GradientLayer

class PINN:


    def __init__(self, network):

        self.network = network
        self.grads = GradientLayer(self.network)

    def build(self):


        # equation input: (t, x)
        TW_eqn = tf.keras.layers.Input(shape=(2,))
        prefactor_eqn = tf.keras.layers.Input(shape=(1,))

        TW_bnd0 = tf.keras.layers.Input(shape=(2,))
        TW_bnd1 = tf.keras.layers.Input(shape=(2,))

        TW_ini = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        C, dC_dT, dC_dW, d2C_dW2 = self.grads(TW_eqn)

        # equation output being zero
        C_eqn = dC_dT - d2C_dW2 - prefactor_eqn*dC_dW
        
        Cbnd0,dC_dT_bnd0,dC_dW_bnd0, d2C_dW2_bnd0 = self.grads(TW_bnd0)
        C_bnd0 = Cbnd0
        Cbnd1,dC_dT_bnd1,dC_dW_bnd1,d2C_dW2_bnd1 = self.grads(TW_bnd1)
        C_bnd1 = Cbnd1
        Cini,dC_dT_ini,dC_dW_ini,d2C_dW2_ini = self.grads(TW_ini)
        C_ini = Cini




        





        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[TW_eqn,prefactor_eqn,TW_bnd0,TW_bnd1,TW_ini], outputs=[C_eqn,C_bnd0,C_bnd1,C_ini])
