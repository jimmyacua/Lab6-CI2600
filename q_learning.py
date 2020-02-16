import gym
import random

Q = {}
alpha = 1
epsilon = 1


def getMaxValue(state):
    """
        Devuelve una tupla con llave y el máximo valor
    """
    global Q
    maxKey = 0  # la primera llave
    maxValue = 0  # el primer valor de la primer llave
    for item in Q[state].items():
        if item[1] > maxValue:
            maxValue = item[1]
            maxKey = item[0]
    return (maxKey, maxValue)


def discretize(obs):
    return int(round(obs[2] * 12)), int(round(obs[3] * 3))


def main():
    global Q
    global epsilon
    global alpha
    env = gym.make("CartPole-v1")
    observation = env.reset()

    action = random.choice([0, 1])  # or env.action_space.sample()
    i = 0
    results = []
    oldObs = observation
    for _ in range(100000):
        env.render()

        observation, reward, done, info = env.step(action)
        # print("Observation", observation)
        # observation
        # [position, velocity, angle, angular velocity]

        # reward
        # 1 if not done
        oldState = discretize(oldObs)
        newState = discretize(observation)
        if oldState not in Q.keys():
            Q[oldState] = {0: 0}
        if action not in Q[oldState].keys():
            Q[oldState].update({action: 0})
        if newState not in Q.keys():
            Q[newState] = {0: 0}

        Q[oldState][action] = (1 - alpha) * Q[oldState][action] + \
                              alpha * (reward + list(getMaxValue(newState))[1])  # se quiere el valor máximo

        #select
        if random.choice([0, 1]) < epsilon:
            action = random.choice([0, 1])
        else:

            action = list(getMaxValue(newState))[0]

        oldObs = observation

        i += 1
        if done:
            alpha = alpha * 0.8888
            epsilon = epsilon * 0.8888
            print(i)
            if i > 100:
                print("!!!!!")
            elif i > 50:
                print("#####")
            results.append(i)
            if len(results) == 100:
                print("avg: ", sum(results) * 1.0 / 100, " alpha: ", alpha, "epsilon: ", epsilon)
                results = []
            i = 0
            observation = env.reset()
            #action = random.choice([0, 1])

    env.close()


if __name__ == "__main__":
    main()
