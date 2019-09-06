def update_env(env, params):
    length, masspole = params
    env.masspole = masspole
    env.total_mass = (masspole + env.masscart)
    env.length = length
    env.polemass_length = (masspole * length)
