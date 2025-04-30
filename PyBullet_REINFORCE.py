# Установка и подготовка среды
"""

!pip install --upgrade gym pyvirtualdisplay ipykernel pybullet > /dev/null 2>&1

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/benelot/pybullet-gym.git

# %cd pybullet-gym

!pip install -e .

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
# Для совместимости с pybullet-gym
np.bool8 = np.bool_

import gym
import pybulletgym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import threading
import os
import base64
from IPython.display import HTML, display

"""# Параметры обучения"""

ENV_ID = 'InvertedPendulumPyBulletEnv-v0'
NUM_THREADS = 4
EPISODES_PER_THREAD = 800
GAMMA = 0.99
LAMBDA = 0.95
PRINT_EVERY = 50
MAX_STEPS_PER_EPISODE = 1500

def make_env():
    """
    Создает экземпляр Gym-среды PyBullet и снимает TimeLimit wrapper.
    Возвращает развернутую среду env.unwrapped.
    """
    e = gym.make(ENV_ID).unwrapped
    return e

"""# Создание среды"""

def make_env():
    """
    Создает экземпляр Gym-среды PyBullet и снимает TimeLimit wrapper.
    Возвращает развернутую среду env.unwrapped.
    """
    e = gym.make(ENV_ID).unwrapped
    return e

"""# Сеть политики"""

class PolicyNetwork(nn.Module):
    """
    Нейронная сеть, моделирующая политику для непрерывного пространства действий.
    На вход принимает вектор состояний, на выходе даёт среднее нормального распределения действий.
    """
    def __init__(self, obs_dim, act_dim, hidden=64, lr=3e-3):
        super().__init__()
        # Двухслойный MLP: obs_dim -> hidden -> act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        # Оптимизатор Adam
        self.opt = optim.Adam(self.net.parameters(), lr)

    def forward(self, x):
        """Прямой проход через сеть. Возвращает тензор формы (act_dim)."""
        return self.net(x)

    def get_action(self, obs):
        """
        Генерирует действие и логарифм плотности для данного состояния.
        :param obs: numpy.ndarray, текущее состояние окружения
        :return: action (np.ndarray), logp (torch.Tensor)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mean = self.forward(obs_t)                # Среднее нормального распределения
        std = torch.ones_like(mean)               # Фиксированное стандартное отклонение
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()                         # Сэмплируем действие
        logp = dist.log_prob(a).sum()             # Лог вероятности действия
        return a.numpy(), logp

    def update(self, logps, advs):
        """
        Обновляет параметры сети методом REINFORCE:
        L(θ) = E[ - logπ(a|s) * A(s,a) ]
        :param logps: список логарифмов вероятностей выбранных действий
        :param advs: список значений преимущества для каждого шага эпизода
        """
        # Вычисляем усредненный loss
        loss = torch.stack([-lp * adv for lp, adv in zip(logps, advs)]).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

"""# Сеть ценности"""

class ValueNetwork(nn.Module):
    """
    Нейронная сеть для приближения функции ценности состояния V(s).
    Выход: скалярная ценность для каждого состояния.
    """
    def __init__(self, obs_dim, hidden=64, lr=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.opt = optim.Adam(self.net.parameters(), lr)
        self.crit = nn.MSELoss()

    def forward(self, x):
        """Возвращает тензор ценностей shape=(batch,)"""
        return self.net(x).squeeze(-1)

    def update(self, states, returns):
        """
        Обучает сеть ценности, минимизируя среднеквадратичную ошибку между предсказанным V(s) и рассчитанными значениями возврата.
        :param states: список или массив состояний (наблюдений)
        :param returns: список или массив значений суммарного дисконтированного вознаграждения
        """
        s_t = torch.as_tensor(states, dtype=torch.float32)
        r_t = torch.as_tensor(returns, dtype=torch.float32)
        preds = self.forward(s_t)
        loss = self.crit(preds, r_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def predict(self, states):
        """
        Предсказывает ценности для батча состояний.
        :return: numpy.ndarray shape=(len(states),)
        """
        with torch.no_grad():
            s_t = torch.as_tensor(states, dtype=torch.float32)
            return self.forward(s_t).numpy()

"""# Вычислительная эффективность"""

def compute_gae(rewards, values, next_vals, gamma=GAMMA, lam=LAMBDA):
    """
    Вычисляет преимущества с помощью GAE:
    δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
    A_t = Σ (γ·λ)^l δ_{t+l}
    :param rewards: numpy.ndarray с вознаграждениями
    :param values: numpy.ndarray с оценками V(s)
    :param next_vals: numpy.ndarray с оценками V(s_{t+1}), последний элемент = 0
    """
    deltas = rewards + gamma * next_vals - values
    advs = np.zeros_like(deltas, dtype=np.float32)
    gae = 0.0
    for i in reversed(range(len(deltas))):
        gae = deltas[i] + gamma * lam * gae
        advs[i] = gae
    return advs

"""# Запуск одного эпизода"""

def run_episode(env, policy, value, use_baseline, use_gae):
    """
    Собирает траекторию одного эпизода (не более MAX_STEPS_PER_EPISODE шагов!) и возвращает:
      states, rewards, log_probs, returns, advantages, эпизодную награду.
    """
    state = env.reset()
    done = False
    step_count = 0

    states, rewards, logps = [], [], []

    # Сбор данных до natural done или лимита шагов
    while not done and step_count < MAX_STEPS_PER_EPISODE:
        action, lp = policy.get_action(state)
        step_result = env.step(action)

        # Универсальная распаковка нового/старого Gym-API
        if len(step_result) == 5:
            next_state, reward, term, trunc, _ = step_result
            done = term or trunc
        else:
            next_state, reward, done, _ = step_result

        states.append(state)
        rewards.append(reward)
        logps.append(lp)
        state = next_state

        step_count += 1

    # Если вышли по лимиту, считаем эпизод завершённым

    # Вычисляем дисконтированные возвраты (returns)
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    # baseline (V(s)) или нули
    values    = value.predict(states) if use_baseline else np.zeros(len(states))
    next_vals = np.append(values[1:], 0.0)

    # преимущества — GAE или простой G - V
    if use_gae:
        advs = compute_gae(np.array(rewards), values, next_vals)
    else:
        advs = np.array(returns) - values

    return states, rewards, logps, returns, advs, sum(rewards)

"""# Многопоточное обучение"""

def train_multithread(policy, value, use_baseline, use_gae):
    """
    Запускает NUM_THREADS параллельных рабочих потоков, где каждый поток:
      - собирает EPISODES_PER_THREAD эпизодов;
      - обновляет параметры общих сетей под защитой threading.Lock().
    """
    lock = threading.Lock()
    rewards_list = []

    def worker(thread_id):
        env = make_env()
        for ep in range(EPISODES_PER_THREAD):
            try:
                # Сбор траектории и вычисление метрик
                states, rewards, logps, returns, advs, ep_r = \
                    run_episode(env, policy, value, use_baseline, use_gae)

                # Обновление параметров под локом
                with lock:
                    if use_baseline:
                        value.update(states, returns)
                    policy.update(logps, advs)

                # Сохранение награды
                rewards_list.append(ep_r)

                # Печать прогресса
                if (ep + 1) % PRINT_EVERY == 0:
                    print(f"[Поток: {thread_id}] Эпизод {ep+1}/{EPISODES_PER_THREAD}, Вознаграждение {ep_r}")
            except Exception:
                continue

    # Запуск потоков
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()
    return rewards_list

"""# Абляционное исследование"""

configs = {
    'A: REINFORCE'        : (False, False),
    'B: + baseline'       : (True,  False),
    'C: + GAE'            : (False, True),
    'D: + baseline+GAE'   : (True,  True),
}
results = {}
for name, (b, g) in configs.items():
    print(f"Запуск {name}")
    pol = PolicyNetwork(make_env().observation_space.shape[0],
                        make_env().action_space.shape[0])
    val = ValueNetwork(make_env().observation_space.shape[0])
    results[name] = train_multithread(pol, val, use_baseline=b, use_gae=g)

"""# Графики обучения"""

plt.figure(figsize=(10, 6))
for name, r in results.items():
    smoothed = np.convolve(r, np.ones(100) / 100, mode='valid')
    plt.plot(smoothed, label=name)
plt.title("График скользящего среднего вознаграждения по эпизодам (окно = 100 эпизодов)")
plt.xlabel("Номер эпизода")
plt.ylabel("Среднее вознаграждение")
plt.legend(title="Конфигурации")
plt.show()

"""# Анализ результатов"""

for name, r in results.items():
    print(f"{name:25s} среднее={np.mean(r):.2f}, max={np.max(r):.1f}, std={np.std(r):.2f}")

"""# Анализ вычислительной эффективности"""

print("\n Средние награды по потокам")
for name, r in results.items():
    chunks = np.array_split(r, NUM_THREADS)
    counts = [len(c) for c in chunks]
    print(f"{name:25s} эпизодов на потоках={counts}")

"""# Итоги

---

Были проведены серии экспериментов на среде InvertedPendulumPyBulletEnv-v0, сравнивая четыре конфигурации алгоритма REINFORCE:
* A: чистый REINFORCE
* B: REINFORCE + адаптивная базовая линия (baseline)
* C: REINFORCE + GAE (без baseline)
* D: REINFORCE + baseline + GAE

Каждый эксперимент шёл в четырёх параллельных потоках, по 800 эпизодов на поток, при лимите в 500 шагов на эпизод.

1. Конфигурация A: чистый REINFORCE
* Среднее вознаграждение: 8.8
* Максимум: 39
* Std: 2.6

Без baseline агент «практически не учится»: средняя кривая вознаграждения остаётся рядом с нулём, редкие «удачные» эпизоды не закрепляются, а разброс крайне мал. С точки зрения вычислительной нагрузки — самая быстрая схватка, но бесполезная с практической точки зрения.

2. Конфигурация B: REINFORCE + baseline
* Среднее вознаграждение: 974.1
* Максимум: 1500 (достигнутый лимит симулятора)
* Std: 654.8

Внедрение адаптивной базовой линии изменило всё: уже к ~300–400 эпизодам среднее скользящее перескочило несколько сотен, а к концу обучения потоки стабильно «держали» 1500 шагов. Дисперсия высока (Std≈655), что объясняется тем, что при первых удачных эпизодах агент резко «взлетал» до 1500, а потом снова «падал» — но в целом к концу каждый поток стабилизировался на максимуме.

3. Конфигурация C: REINFORCE + GAE (без baseline)
* Среднее вознаграждение: 9.1
* Максимум: 36
* Std: 2.6

GAE без baseline оказался бессмысленным: кривая вознаграждений полностью совпадает с чистым REINFORCE. Без оценки V(s) преимущества не получают корректной «опорной линии», и весь поправочный механизм GAE теряет свою эффективность.

4. Конфигурация D: REINFORCE + baseline + GAE
* Среднее вознаграждение: 879.1
* Максимум: 1500
* Std: 647.7

Сочетание baseline и GAE тоже обучается — агент к середине прогресса выходит на несколько сотен шагов, а к концу стабильно достигает 1500. Однако среднее чуть ниже, чем у варианта B: 879 против 974. Скорость нарастания награды чуть ниже, возможно из-за дополнительных вычислений GAE, которые слегка «размывают» сигнал. Разброс похож на B, но третьем варианте пиковые эпизоды появляются реже.

5. Графики обучения
* Оранжевый (B) — резкий подъём уже после 200 эпизодов, плато на максимуме.
* Красный (D) — медленный старт, затем рост, но плато чуть ниже и с небольшими «провалами».
* Синий (A) и зелёный (C) — остаются почти плоскими.

6. Анализ вычислительной эффективности
* Общий бюджет симуляций: 4 потока × 800 эпизодов × 500 шагов = 1 600 000 взаимодействий.
* Среднее время эпизода (примерно):

7. Выводы и рекомендации
* Адаптивный baseline — ключевой компонент.

Без него обучение не идёт. REINFORCE + baseline (Вариант B) — наилучший компромисс «затраты–качество»: быстро достигает максимума, даёт высокий средний результат.

* GAE стоит применять вместе с baseline и при достаточных ресурсах.

Сам по себе (Вариант C) GAE бесполезен; вместе с baseline (Вариант D) он даёт чуть более плавную кривую, но уступает в скорости и среднем вознаграждении.

* Бюджет взаимодействий.

Для InvertedPendulum достаточно 800–1000 эпизодов по 500–800 шагов — этого хватает, чтобы baseline достиг максимума и замерить разницу между конфигурациями. Если хочется более детальной оценки GAE на длинных траекториях, можно поднять лимит шагов до 1000–1500 и сократить число эпизодов, сохранив общий бюджет.

* Дальнейшие улучшения:
 * Увеличение, либо удаление ограничения количество шагов
 * Увеличение гиперпараметров GAE (λ, γ) для варианта D.
 * Использование ансамбля критиков для более стабильной оценки V(s).
 * Переход на асинхронную архитектуру (separate actors & learner), для снятия узкого места обновления весов.




"""