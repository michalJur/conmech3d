import optax
import jax.numpy as jnp
import jax

# @jax.jit() #static_argnames=['fun', 'opt', 'max_iter', 'tol'])
def run_opt(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = optax.tree_utils.tree_get(state, 'count')
    grad = optax.tree_utils.tree_get(state, 'grad')
    err = optax.tree_utils.tree_l2_norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state


def fun(w):
  return jnp.sum(100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2)

opt = optax.lbfgs()
init_params = jnp.zeros((8,))
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree_utils.tree_l2_norm(jax.grad(fun)(init_params)):.2e}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)

print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree_utils.tree_l2_norm(jax.grad(fun)(final_params)):.2e}'
)