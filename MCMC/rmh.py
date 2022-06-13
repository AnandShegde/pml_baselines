import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import os

def rmh_inference_loop(dist_numerator,sigma_rmh,params,num_samples):
    rmh = blackjax.rmh(dist_numerator, sigma= sigma_rmh)
    # initial = jnp.array([10.4,11.4])
    initial_state = rmh.init(params)
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states
    rng_key = jax.random.PRNGKey(0)
    _, rng_key = jax.random.split(rng_key)
    states = inference_loop(rng_key, rmh.step, initial_state, num_samples)
    return states

def plot_rmh(states,burnin,contains_arrays=False):
    for key in states.position:
        if jnp.ndim(states.position[key])==1:
            fig, (ax,ax1) = plt.subplots(1,2,figsize=(13,5))
            ax.plot(states.position[key])
            ax.set_title('trace plot')
            ax.axvline(x=burnin, c="tab:red")
            ax1.hist(states.position[key],density=True)
            sns.kdeplot(states.position[key],ax=ax1)
            ax1.set_title("histogram")
            plt.suptitle(key,fontsize=20)
            try:
                plt.savefig('figures/'+key+'.jpeg')
            except:
                os.mkdir('figures')
                plt.savefig('figures/'+key+'.jpeg')
            plt.show()
        
        else:
            for i in range(states.position[key].shape[1]):
                fig, (ax,ax1) = plt.subplots(1,2,figsize=(8,3))
                ax.plot(states.position[key][:,i])
                ax.set_title('trace plot')
                ax.axvline(x=burnin, c="tab:red")
                ax1.hist(states.position[key][burnin:,i],density=True)
                sns.kdeplot(states.position[key][burnin:,i],ax=ax1)
                ax1.set_title("histogram")
                plt.suptitle(key+str(i),fontsize=20)
                try:
                    plt.savefig('figures/'+key+str(i)+'.jpeg')
                except:
                    os.mkdir('figures')
                    plt.savefig('figures/'+key+str(i)+'.jpeg')
                plt.show()

