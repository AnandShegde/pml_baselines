import jax
import optax


def train_model(model, data, optimizer, n_epochs, seed):
    params = model.init(seed)
    value_and_grad_fn = jax.value_and_grad(model.loss_fun)
    state = optimizer.init(params)

    @jax.jit
    def one_step(params_and_state, xs):
        params, state = params_and_state
        loss, grads = value_and_grad_fn(params, data)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), loss

    params_and_states, losses = jax.lax.scan(
        one_step, (params, state), xs=None, length=n_epochs
    )
    return params_and_states[0], losses
