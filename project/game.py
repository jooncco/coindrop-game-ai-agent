import pygame
import random

class Game:
    def __init__(self, show_game=True):
        self.score = 0
        self.global_counter = 0
        self.counter = 200
        self.show_game = show_game
        self.width, self.height = (640, 480)

        self.timer_location = [560, 20]
        self.score_location = [10, 20]

        # 0: nothing, 1: left, 2: right
        self.basket_pos = [64*4, 400]
        self.basket_action = 0

        self.all_item_info = [[1, self._gen_random_start_point(), 0]]

        if self.show_game:
            self._prepare_display()

    def get_action_num():
        return 3

    def _gen_random_start_point(self):
        return 48 * random.randint(0, 11) + 32

    def _conv_item_info(self):
        new_data = []
        for item in self.all_item_info:
            new_data.append([item[0], (item[1] - 32) // 48, item[2] // 48])
        return new_data

    def _prepare_display(self):
        pygame.init()
        self.screen=pygame.display.set_mode((self.width, self.height))

        self.back = pygame.image.load("resources/background.png")
        self.basket = pygame.image.load("resources/basket.png")
        self.basket = pygame.transform.scale(self.basket, (128, 90))

        # coin: 1, time: 2
        self.item_coin = pygame.image.load("resources/item_coin.png")
        self.item_coin = pygame.transform.scale(self.item_coin, (48, 48))

        self.item_time = pygame.image.load("resources/item_time.png")
        self.item_time = pygame.transform.scale(self.item_time, (48, 48))

        self.start_ticks=pygame.time.get_ticks()
        pygame.time.set_timer(pygame.USEREVENT, 75)
        self.font = pygame.font.SysFont('Segoe UI', 30)


    def get_count(self):
        count_data = {1:0, 2:0}
        for item in self.all_item_info:
            count_data[item[0]] += 1
        return count_data

    def is_in_basket(self, pos):
        return pos >= self.basket_pos[0] and pos <= self.basket_pos[0] + 128

    def reset(self):
        self.__init__(self.show_game)
        return (self.counter <= 0, self.counter, self.score, (self.basket_pos[0] // 64, self._conv_item_info()))

    def step(self, action, user_input=False):

        # move and calculate score and counter phase
        if self.counter > 0:
            self.counter -= 1
            if self.basket_action == 1:
                if self.basket_pos[0] > 0:
                    self.basket_pos[0] -= 64

            elif self.basket_action == 2:
                if self.basket_pos[0] < 512:
                    self.basket_pos[0] += 64

            for i in range(len(self.all_item_info)):
                self.all_item_info[i][2] += 48

            if (self.counter % 3 == 0):
                self.all_item_info.append([1, self._gen_random_start_point(), 0])

            if (self.get_count()[2] == 0 and self.counter < 500 and self.score < 300000 and self.counter % 10 == 0 and random.randint(0, 10) < 3):
                self.all_item_info.append([2, self._gen_random_start_point(), 0])

            # print(self.global_counter, self.counter, self.score, self.basket_action, self.basket_pos[0] // 64, self._conv_item_info())

        # action input phase
        if self.show_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    exit(0)

                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()
                    pass

                if user_input:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.basket_action = 1
                        elif event.key == pygame.K_RIGHT:
                            self.basket_action = 2
                        else:
                            self.basket_action = 0

                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT:
                            if self.basket_action == 1:
                                self.basket_action = 0
                        elif event.key == pygame.K_RIGHT:
                            if self.basket_action == 2:
                                self.basket_action = 0
                        else:
                            self.basket_action = 0

            if not user_input:
                self.basket_action = action
        else:
            self.basket_action = action

        # calculate next state phase
        if self.counter > 0:
            new_all_item_info = []
            for item_info in self.all_item_info:
                item_type, item_pos = (item_info[0], item_info[1:])
                is_bottom = item_pos[1] == 480
                if item_type == 1:
                    if is_bottom:
                        if self.is_in_basket(item_pos[0]):
                            self.score += 500
                        else:
                            self.score -= 150
                if item_type == 2:
                    if is_bottom and self.is_in_basket(item_pos[0]):
                        self.counter += 100

                if not is_bottom:
                    new_all_item_info.append(item_info)

            self.all_item_info = new_all_item_info
            del new_all_item_info


        if self.show_game:
            self.screen.fill((0,0,0))
            self.screen.blit(self.back, (0,0))

            desc_font = pygame.font.SysFont('Segoe UI', 25)
            self.screen.blit(desc_font.render("q: exit / r: reset", True, (255, 255, 255)), (10, 440))

            for item_info in self.all_item_info:
                item_type, item_pos = (item_info[0], item_info[1:])
                is_bottom = item_pos[1] == 432
                if not is_bottom or not self.is_in_basket(item_pos[0]):
                    if item_type == 1:
                        self.screen.blit(self.item_coin, item_pos)
                    if item_type == 2:
                        self.screen.blit(self.item_time, item_pos)

            self.screen.blit(self.basket, self.basket_pos)
            time_text = "{:>4d} ".format(self.counter)
            self.screen.blit(self.font.render(time_text, True, (255, 255, 255), (75, 107, 32)), self.timer_location)

            score_text = "{:>10d} ".format(self.score)
            self.screen.blit(self.font.render(score_text, True, (255, 255, 255), (0, 131, 143)), self.score_location)

            pygame.display.flip()

            pygame.time.delay(75)
        self.global_counter += 1

        # return is_done, counter, score, state
        return (self.counter <= 0, self.counter, self.score, (self.basket_pos[0] // 64, self._conv_item_info()))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", help="1: show game in GUI, 0: only print the state",
                    type=int, default=1)
    parser.add_argument("-p", "--playmode", help="1: play the game (--show must be 1), 0: run your agent",
                    type=int, default=1)
    parser.add_argument("-m", "--model", help="your model file",
                    type=str, default="model_q.pkl")

    args = parser.parse_args()

    show = args.show is 1
    play_mode = args.playmode is 1
    model_fn = args.model
    if not play_mode:
        print("load model: ", model_fn)

    if not show and play_mode:
        print("--show must be 1 when --playmode is 1")
        exit(0)

    if not play_mode:
        from q_learning import load_q, get_action
        Q, n_ep, params  = load_q(model_fn)

    g = Game(show)
    is_done, counter, score, game_info = g.reset()

    print_flag = True
    while not is_done or show:
        if play_mode:
            action = random.randint(0,2)
        else:
            action = get_action(Q, counter, score, game_info, params)

        is_done, counter, score, game_info = g.step(action, play_mode)
        if print_flag:
            print(is_done, counter, score, game_info)
        print_flag = not is_done

if __name__ == "__main__":
    main()
