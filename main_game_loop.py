# main_game_loop.py (イベント処理一元化バージョン)
import pygame
import sys
import os
from game_py.MyGameEnv import MyGameEnv # 作成したゲーム環境
from stable_baselines3 import PPO # 学習済みモデルを読み込むため
import numpy as np # obsの初期化に必要になる可能性

def resource_path(relative_path):
    """ .exe化した後でも、リソースファイルへの正しいパスを取得する """
    try:
        # PyInstallerが作成した一時フォルダのパスを取得
        base_path = sys._MEIPASS
    except Exception:
        # 通常のPython環境で実行している場合
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# --- 定数 ---

# モデルファイルとハイスコアファイルのパスを新しい関数で取得する
MODEL_FILENAME = resource_path("ppo_rl_shooter.zip")
HIGHSCORE_FILENAME = resource_path("highscore.txt")

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 120 #初期値60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


MODEL_FILENAME = "ppo_rl_shooter.zip"

# --- ゲーム状態 ---
STATE_TITLE_SCREEN = 0
STATE_PLAYING_GAME = 1
STATE_GAME_OVER = 2

# --- フォントの初期化 ---
pygame.font.init()
try:
    TITLE_FONT = pygame.font.Font(None, 74)
    SCORE_FONT = pygame.font.Font(None, 50)
    SMALL_FONT = pygame.font.Font(None, 36)
except pygame.error:
    TITLE_FONT = pygame.font.SysFont("arial", 70)
    SCORE_FONT = pygame.font.SysFont("arial", 45)
    SMALL_FONT = pygame.font.SysFont("arial", 30)

def load_high_score():
    try:
        # with open("highscore.txt", "r") as f: #  元のコード
        with open(HIGHSCORE_FILENAME, "r") as f: #  修正後のコード
            return int(f.read())
    except (FileNotFoundError, ValueError):
        return 0

def save_high_score(score):
    # with open("highscore.txt", "w") as f: # ← 元のコード
    with open(HIGHSCORE_FILENAME, "w") as f: # ← 修正後のコード
        f.write(str(score))

def draw_text(surface, text, font, color, center_x, center_y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(center_x, center_y))
    surface.blit(text_surface, text_rect)

# --- メインループ ---
def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RL シューティングゲーム")
    clock = pygame.time.Clock()

    current_state = STATE_TITLE_SCREEN
    game_env = None
    model = None
    obs = None # game_env.reset() から取得する観測情報

    # モデルの読み込み
    if os.path.exists(MODEL_FILENAME):
        try:
            # MyGameEnv の action_space, observation_space を使ってロードできるか試す
            # ダミー環境を作成して、そのスペース情報をモデルロードに使う
            # これはモデルが環境のスペース情報を必要とする場合があるため
            # MyGameEnv が引数なしで初期化できることを確認してください
            temp_env_for_load = MyGameEnv()
            model = PPO.load(MODEL_FILENAME, env=temp_env_for_load)
            temp_env_for_load.close() # ダミー環境はすぐに閉じる
            print(f"モデル {MODEL_FILENAME} をロードしました。")
        except Exception as e:
            print(f"モデルのロードに失敗しました: {e}")
            model = None
    else:
        print(f"モデルファイル {MODEL_FILENAME} が見つかりません。RLエージェントによる敵の制御はありません。")

    last_score = 0
    high_score = load_high_score()

    player_actions_this_frame = {'left': False, 'right': False, 'shoot': False}

    running = True
    while running:
        # --- 1. イベント処理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                if game_env:
                    game_env.user_quit = True # 環境に通知

            if current_state == STATE_PLAYING_GAME:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player_actions_this_frame['left'] = True
                    elif event.key == pygame.K_RIGHT: # 左と右は同時押しを考慮して elif
                        player_actions_this_frame['right'] = True
                    if event.key == pygame.K_UP:
                        player_actions_this_frame['shoot'] = True
                    if event.key == pygame.K_ESCAPE:
                        print("ゲームプレイ中にESCキー: タイトルに戻ります。")
                        current_state = STATE_TITLE_SCREEN
                        if game_env:
                            game_env.close()
                            game_env = None
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        player_actions_this_frame['left'] = False
                    if event.key == pygame.K_RIGHT:
                        player_actions_this_frame['right'] = False
                    # 'shoot' は KEYDOWN のみでトリガーするので、KEYUPでのリセットは不要
                    # (押しっぱなしで連射する場合はロジック変更が必要)

            elif current_state == STATE_TITLE_SCREEN:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        print("タイトル画面からゲーム開始")
                        current_state = STATE_PLAYING_GAME
                        if game_env: # 古い環境がもしあれば閉じる
                            game_env.close()
                        #game_env = MyGameEnv(render_mode="human")
                        game_env = MyGameEnv(screen_surface=screen, render_mode="human") # ここ！
                        obs, info = game_env.reset()
                        player_actions_this_frame = {'left': False, 'right': False, 'shoot': False} # アクション状態をリセット
                    if event.key == pygame.K_ESCAPE:
                        running = False

            elif current_state == STATE_GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        current_state = STATE_TITLE_SCREEN
                    if event.key == pygame.K_ESCAPE:
                        running = False
        
        # --- 2. ゲームロジックの更新 ---
        if current_state == STATE_PLAYING_GAME:
            if game_env:
                # RLエージェントによる敵のアクション決定
                enemy_action = 1 # デフォルトアクション（モデルがない、または何らかの理由でエラーの場合）
                                 # MyGameEnvのアクションスペース(0:左, 1:停止, 2:右) の「停止」
                if model and obs is not None:
                    try:
                        enemy_action, _states = model.predict(obs, deterministic=True)
                    except Exception as e:
                        print(f"model.predictでエラー: {e}。デフォルトアクションを使用します。")
                        # obs が不正な場合などにエラーが起きうる
                
                # 環境を1ステップ進める (プレイヤーのアクションを渡す)
                next_obs, reward, terminated, truncated, info = game_env.step(enemy_action, player_actions_this_frame)
                obs = next_obs # 次の観測を更新

                # 単発の射撃アクションをリセット
                player_actions_this_frame['shoot'] = False

                last_score = info.get('score', last_score) # スコアを更新

                if terminated or truncated or info.get("user_quit", False):
                    print("ゲームオーバー！")
                    current_state = STATE_GAME_OVER
                    if last_score > high_score:
                        high_score = last_score
                        save_high_score(high_score)
                    # game_env.close() はゲームオーバー画面からタイトルに戻るときや、
                    # 次のゲームを開始するときに呼ばれるので、ここでは呼ばない方が良い場合もある。
                    # (スコア表示などに環境内の情報を使わないなら即時closeでも可)
            else: # game_env が None の場合 (通常は発生しないはず)
                print("エラー: game_envがNoneの状態でSTATE_PLAYING_GAMEです。タイトルに戻ります。")
                current_state = STATE_TITLE_SCREEN

        # --- 3. 画面描画 ---
        #screen.fill(BLACK) # 毎フレーム背景をクリア

        if current_state == STATE_TITLE_SCREEN:
            draw_text(screen, "RL シューティング", TITLE_FONT, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
            draw_text(screen, "Press ENTER to Start", SCORE_FONT, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            draw_text(screen, f"High Score: {high_score}", SMALL_FONT, GREEN, SCREEN_WIDTH // 2, SCREEN_HEIGHT * 3 // 4)

        elif current_state == STATE_PLAYING_GAME:
            if game_env:
                # MyGameEnvのrenderメソッドはstep内で呼ばれるか、ここで明示的に呼ぶ
                # MyGameEnv.step内でrender_mode="human"の場合にrender()を呼んでいるなら、ここでは不要。
                # もしstep内でrenderが呼ばれない設計なら、ここで game_env.render() を呼ぶ。
                # (現在のMyGameEnv.step()は内部でrenderを呼んでいるので、ここでは不要)
                pass # 描画はMyGameEnv.render() -> pygame.display.flip() (メインループ末尾) で行われる

        elif current_state == STATE_GAME_OVER:
            draw_text(screen, "GAME OVER", TITLE_FONT, RED, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
            draw_text(screen, f"Your Score: {last_score}", SCORE_FONT, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            draw_text(screen, f"High Score: {high_score}", SCORE_FONT, GREEN, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
            draw_text(screen, "Press ENTER to Return to Title", SMALL_FONT, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT * 3 // 4 + 20)

        pygame.display.flip() # 画面全体を更新 (1フレームに1回だけ呼ぶ)
        clock.tick(FPS) # FPS制御

    # --- ループ終了後の処理 ---
    if game_env:
        game_env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game_loop()