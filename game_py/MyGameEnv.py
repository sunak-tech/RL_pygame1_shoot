# MyGameEnv.py

import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import numpy as np
# import os # os は Sprite Classes より前で使われていないので、ここでなくてもOK

# Game Constants (変更なし)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_SIZE = 50
ENEMY_SIZE = 30
PLAYER_COLOR = (0, 128, 255)
ENEMY_COLOR = (255, 0, 0)
BULLET_COLOR = (255, 255, 0)
BULLET_SPEED = 10
PLAYER_SPEED = 5
BACKGROUND_COLOR = (0, 0, 0)
MAX_OBSERVED_ENEMIES = 3
OBS_ENEMY_DATA_SIZE = 4

# --- Sprite Classes --- (変更なし)
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([PLAYER_SIZE, PLAYER_SIZE])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 50
        self.speed_x = 0
        self.shoot_cooldown = 0
        self.shoot_delay = 15
        

    def update(self):
        self.rect.x += self.speed_x
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def attempt_shoot(self):
        if self.shoot_cooldown == 0:
            self.shoot_cooldown = self.shoot_delay
            return True
        return False

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([ENEMY_SIZE, ENEMY_SIZE])
        self.image.fill(ENEMY_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - ENEMY_SIZE)
        self.rect.y = random.randrange(-ENEMY_SIZE * 3, -ENEMY_SIZE)
        self.base_downward_speed = random.uniform(1.5, 3.5)
        self.horizontal_speed = 5.0

    def update(self, agent_action):
        self.rect.y += self.base_downward_speed
        move_x = 0
        if agent_action == 0: # 左へ
            move_x = -self.horizontal_speed
        elif agent_action == 2: # 右へ
            move_x = self.horizontal_speed
        self.rect.x += move_x
        if self.rect.left < 0: self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH: self.rect.right = SCREEN_WIDTH
        if self.rect.top > SCREEN_HEIGHT:
            self.rect.x = random.randrange(0, SCREEN_WIDTH - ENEMY_SIZE)
            self.rect.y = random.randrange(-ENEMY_SIZE * 2, -ENEMY_SIZE)

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([10, 20])
        self.image.fill(BULLET_COLOR)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y

    def update(self):
        self.rect.y -= BULLET_SPEED
        if self.rect.bottom < 0:
            self.kill()

class MyGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120} # FPSはこちらで定義しても良い.初期値60

    # ★★★ __init__ を修正 ★★★
    def __init__(self, screen_surface=None, render_mode=None, initial_enemies=5, max_episode_steps=2000):
        super().__init__()
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.render_mode = render_mode
        self.initial_enemies = initial_enemies
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(3) # 0:左, 1:停止, 2:右
        obs_shape = 2 + MAX_OBSERVED_ENEMIES * OBS_ENEMY_DATA_SIZE
        # 観測値が必ずしも[-1, 1]に収まらない場合や、正規化を別途行う場合は np.inf を使う
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        # Pygameモジュール自体の初期化 (フォントなどを使う前に必要)
        pygame.init() # 複数回呼んでも安全だが、main_game_loop側で1回呼んでいれば不要な場合も
        try:
            self.font = pygame.font.Font(None, 36)
        except pygame.error:
            self.font = pygame.font.SysFont("arial", 30)
        # self.clock = pygame.time.Clock() # clock は main_game_loop で管理

        if self.render_mode == "human":
            if screen_surface:
                self.screen = screen_surface # main_game_loop から渡された screen を使用
                self._created_screen_independently = False
            else:
                # screen_surface が渡されない場合は、MyGameEnv が自分でウィンドウを作る (主にテスト用)
                print("MyGameEnv (human mode): screen_surface not provided, creating own display.")
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("RL Shooter - MyGameEnv (Self-Managed Window)")
                self._created_screen_independently = True # 自分で作ったフラグ
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self._created_screen_independently = True # rgb_arrayでも内部でSurfaceを作る
        else: # render_mode is None
            self.screen = None
            self._created_screen_independently = False


        # Game state variables (変更なし)
        self.player = None
        self.enemies = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.score = 0
        self.level = 1
        self.current_step = 0
        self.user_quit = False

    # _initialize_pygame メソッドは不要になるか、役割が変わる
    # def _initialize_pygame(self):
    #     # この内容は __init__ に統合されたか、不要になった

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pygameの初期化やscreenの設定は__init__で行われる
        if self.render_mode == "human" and self.screen is None:
             # __init__で screen が設定されなかった場合のエラーやフォールバック処理
             # (例えば、gane_venv_self.py から直接使われた場合など)
            print("MyGameEnv.reset(): render_mode='human' で self.screen が None です。独立してディスプレイを初期化します。")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("RL Shooter - MyGameEnv (Reset Fallback)")
            self._created_screen_independently = True
        elif self.screen is None and self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self._created_screen_independently = True


        self.player = Player()
        self.enemies.empty()
        self.bullets.empty()
        self.all_sprites.empty()
        self.all_sprites.add(self.player)
        self._spawn_enemies(self.initial_enemies + (self.level - 1) * 2)
        self.score = 0
        self.level = 1
        self.current_step = 0
        self.user_quit = False
        return self._get_obs(), {}

    def _spawn_enemies(self, number_of_enemies): # 変更なし
        for _ in range(number_of_enemies):
            enemy = Enemy()
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)

    def _get_obs(self): # 変更なし (ただしobservation_spaceのlow/highと合わせる必要あり)
        obs_list = []
        if self.player is None: # playerがまだ初期化されていない場合（reset前など）
            # obs_shapeに合わせたダミーの値を返すか、エラーを出す
            # ここでは一旦、エラーを避けるためにダミー値を返す（理想的ではない）
            print("Warning: _get_obs called when player is None. Returning zero observation.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs_list.append(np.clip(self.player.rect.centerx / self.screen_width * 2 - 1, -1.0, 1.0))
        obs_list.append(np.clip(self.player.rect.centery / self.screen_height * 2 - 1, -1.0, 1.0))

        sorted_enemies = sorted(self.enemies.sprites(), key=lambda e: np.hypot(e.rect.centerx - self.player.rect.centerx, e.rect.centery - self.player.rect.centery))
        for i in range(MAX_OBSERVED_ENEMIES):
            if i < len(sorted_enemies):
                enemy = sorted_enemies[i]
                obs_list.append(np.clip(enemy.rect.centerx / self.screen_width * 2 - 1, -1.0, 1.0))
                obs_list.append(np.clip(enemy.rect.centery / self.screen_height * 2 - 1, -1.0, 1.0))
                obs_list.append(np.clip((self.player.rect.centerx - enemy.rect.centerx) / self.screen_width, -1.0, 1.0))
                obs_list.append(np.clip((self.player.rect.centery - enemy.rect.centery) / self.screen_height, -1.0, 1.0))
            else:
                obs_list.extend([0.0] * OBS_ENEMY_DATA_SIZE)
        return np.array(obs_list, dtype=np.float32)


    def _get_reward(self, terminated, enemies_killed_this_step):
        reward = 0.0

        # 1. プレイヤーと衝突しゲームオーバーさせる報酬 (逆転)
        if terminated: # プレイヤーが敵に衝突してゲームオーバーになった場合
            reward += 100.0 # 大きな正の報酬を与える
        else:
            reward -= 0.1 # ゲームが続いている間はわずかな負の報酬 (積極的にゲームを終わらせるため)

        # 2. プレイヤーの攻撃をよける報酬 (逆転)
        # 敵が倒されることに対して負の報酬を与える
        reward -= enemies_killed_this_step * 20.0 # 敵が倒されると負の報酬

        # その他の調整 (例: プレイヤーに近いほど報酬など)
        #ここに、敵がプレイヤーにどれだけ近づいたか、などを考慮した報酬を追加することもできます。
        #例:
        if self.player and self.enemies:
           min_dist_to_player = float('inf')
           for enemy in self.enemies:
               dist = np.hypot(enemy.rect.centerx - self.player.rect.centerx, enemy.rect.centery - self.player.rect.centery)
               min_dist_to_player = min(min_dist_to_player, dist)
           # 距離が近いほど報酬が大きくなるように調整 (正規化が必要)
           reward += max(0, (SCREEN_HEIGHT - min_dist_to_player) / SCREEN_HEIGHT * 5) # 距離が近いほど報酬

        return reward

    # ★★★ step メソッドの引数を修正し、内部のイベント処理を削除 ★★★
    def step(self, enemy_agent_action, player_actions=None): # player_actions を受け取る
        #print("MyGameEnv.step() called")  # デバッグ用
        if self.user_quit:
             return self._get_obs(), 0, False, True, {"user_quit": True}

        self.current_step += 1
        enemies_killed_this_step = 0
        player_shot_this_step = False

        # --- Player action from player_actions dictionary ---
        if self.player: # playerオブジェクトが存在するか確認
            if player_actions:
                if player_actions.get('left'):
                    self.player.speed_x = -PLAYER_SPEED
                elif player_actions.get('right'):
                    self.player.speed_x = PLAYER_SPEED
                else:
                    self.player.speed_x = 0

                if player_actions.get('shoot'):
                    if self.player.attempt_shoot():
                        bullet = Bullet(self.player.rect.centerx, self.player.rect.top)
                        self.bullets.add(bullet)
                        self.all_sprites.add(bullet)
                        player_shot_this_step = True
            else: # player_actions が None の場合
                self.player.speed_x = 0
        else:
            print("Warning: Player object is None in step method.")


        # --- Update Game State ---
        if self.player: self.player.update() # Player が None でないことを確認
        self.bullets.update()
        for enemy in self.enemies:
            enemy.update(enemy_agent_action)

        # --- Collision Detection ---
        terminated = False
        if self.player and pygame.sprite.spritecollide(self.player, self.enemies, True):
            print(f"！！！ Player COLLIDED with enemies. Terminating episode. ！！！")
            terminated = True

        killed_dict = pygame.sprite.groupcollide(self.enemies, self.bullets, True, True)
        for _ in killed_dict: # Python 3.8+ では for enemy_hit in killed_dict: の方が明確
            self.score += 50
            enemies_killed_this_step += 1

        # --- Leveling and Enemy Respawning ---
        if not self.enemies and not terminated:
            self.level += 1
            self.score += 100
            self._spawn_enemies(self.initial_enemies + (self.level - 1) * 2)

        # --- Termination and Truncation ---
        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True

        observation = self._get_obs()
        reward = self._get_reward(terminated, enemies_killed_this_step)
        info = {"score": self.score, "level": self.level, "user_quit": self.user_quit}

        if self.render_mode == "human" and not self.user_quit :
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self): # ★★★ flip と tick を削除済 ★★★
        if self.screen is None or self.user_quit: return None

        self.screen.fill(BACKGROUND_COLOR)
        self.all_sprites.draw(self.screen)

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        level_text = self.font.render(f"Level: {self.level}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(level_text, (10, 40))

        #pygame.display.flip() #削除
        #self.clock.tick(self.metadata["render_fps"]) # 削除

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            # pygame.display.quit() # ★コメントアウトまたは削除
            # pygame.quit()         # ★コメントアウトまたは削除
            # 以下は残しても良い
            if hasattr(self, '_created_screen_independently') and self._created_screen_independently:
                # MyGameEnvが独自にウィンドウを作った場合のみdisplay.quitを呼ぶことも考えられるが、
                # main_game_loop.pyで最終的にpygame.quit()が呼ばれるので、ここでは何もしないのが安全。
                pass
            self.screen = None
            print("MyGameEnv closed.") # メッセージ変更の提案
        self.user_quit = True