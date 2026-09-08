import random
import pygame
import torch

train_data = "/home/ordn/Documents/ordn_projects/car_game_ai/training_data.txt"

ACTION_STAND_STILL = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4

DIRECTION_DOWN = 21
DIRECTION_UP = 23
MAX_OBSERVED_OBSTACLES = 5

class Game:

  def __init__(self, ai=None):

    self.ai = ai
    self.played = False
    pygame.init()
    self.screen_width = 800
    self.screen_height = 600

    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
    pygame.display.set_caption("Car Game AI")

    self.clock = pygame.time.Clock()
    self.running = True
    self.fps = 60
    self.car_width = 40
    self.car_height = 70
    self.player_velocity = 5
    self.min_obstacle_velocity = 2
    self.max_obstacle_velocity = 8
    self.obstacle_spawn_range = 200
    self.obstacle_spawn_threshold = 2
    self.font = pygame.font.Font(None,30)

    self.player_car_image = (self.create_car_image(body_color=(0, 180, 0), detail_color=(0, 255, 0)))
    self.obstacle_car_images = [
      self.create_car_image(body_color=(180, 0, 0),detail_color=(255, 0, 0)),
      self.create_car_image(body_color=(0, 0, 180),detail_color=(0, 100, 255)),
      self.create_car_image(body_color=(180, 180, 0),detail_color=(255, 255, 0)),
      self.create_car_image(body_color=(180, 0, 180),detail_color=(255, 0, 255))]

  def create_car_image(self,body_color,detail_color):
    car_image = pygame.Surface((self.car_width,self.car_height),pygame.SRCALPHA)
    pygame.draw.rect(car_image,body_color,(3,4,self.car_width - 6,self.car_height - 8),border_radius=8)
    pygame.draw.rect(car_image,(140, 200, 255),(8,15,self.car_width - 16,20),border_radius=4)
    pygame.draw.rect(car_image,detail_color,(7,5,self.car_width - 14,8),border_radius=3)

    wheel_color = (20,20,20)

    pygame.draw.rect(car_image,wheel_color,(0,10,5,15))
    pygame.draw.rect(car_image,wheel_color,(self.car_width - 5,10,5,15))
    pygame.draw.rect(car_image,wheel_color,(0,self.car_height - 25,5,15))
    pygame.draw.rect(car_image,wheel_color,(self.car_width - 5,self.car_height - 25,5,15))

    return car_image

  def initialize_game(self):

    self.game_map = (self.generate_game_map())
    self.current_lane = 0
    self.car_position = [int(self.game_map['lanes'][self.current_lane]['car_x']),300]
    self.obstacles = []
    self.data = []
    self.score = 0
    self.collided = False
    self.start_time = (pygame.time.get_ticks())
    self.survival_time = 0.0

  def generate_game_map(self):

    self.hwroad_height = (self.screen_height)
    road_left = 300
    outer_line_width = 5
    lane_line_width = 5
    median_width = 7
    lane_width = 50

    left_outer_line = (road_left)
    first_inner_line = (left_outer_line + outer_line_width + lane_width)
    median_left = (first_inner_line + lane_line_width + lane_width)
    median_right = (median_left + median_width)
    second_inner_line = (median_right + lane_width)
    right_outer_line = (second_inner_line + lane_line_width + lane_width + outer_line_width)

    lane_1_left = (left_outer_line + outer_line_width)
    lane_1_right = (first_inner_line)
    lane_2_left = (first_inner_line + lane_line_width)
    lane_2_right = (median_left)
    lane_3_left = (median_right)
    lane_3_right = (second_inner_line)
    lane_4_left = (second_inner_line + lane_line_width)
    lane_4_right = (right_outer_line - outer_line_width)

    def get_car_x(lane_left,lane_right):

      lane_center = (lane_left + lane_right) / 2
      car_x = (lane_center - self.car_width / 2)

      return int(car_x)

    return {
      'road_left': int(road_left),
      'road_right': int(right_outer_line),
      'left_outer_line': int(left_outer_line),
      'first_inner_line': int(first_inner_line),
      'median_left': int(median_left),
      'median_right': int(median_right),
      'second_inner_line': int(second_inner_line),
      'right_outer_line': int(
        right_outer_line),

      'lanes': [{'name': 'left_outer',
          'car_x': int(get_car_x(lane_1_left,lane_1_right)),
          'direction': 'down'},
        {'name': 'left_inner',
          'car_x': int(get_car_x(lane_2_left,lane_2_right)),
          'direction': 'down'},
        {'name': 'right_inner',
          'car_x': int(get_car_x(lane_3_left,lane_3_right)),
          'direction': 'up'},
        {'name': 'right_outer',
          'car_x': int(get_car_x(lane_4_left,lane_4_right)),
          'direction': 'up'}]}
  
  def ai_response(self):

    inn = torch.tensor([self.data[-1][:-1]], dtype=torch.float32) if self.played else torch.tensor(self.ai.get_deflt(), dtype=torch.float32)
    self.played = True

    if self.ai.fed >= self.ai.batch:
      self.ai.initialize_game()

    ix = self.ai.sample_game(inn)
    self.ai.fed += 1

    return ix

  def handle_events(self):

    horizontal_action = (ACTION_STAND_STILL)

    for event in pygame.event.get():

      if event.type == pygame.QUIT:
        self.running = False

      elif event.type == pygame.KEYDOWN:

        if type(self.ai) == type(None):

          if event.key == pygame.K_LEFT:
            horizontal_action = ACTION_LEFT

          elif event.key == pygame.K_RIGHT:
            horizontal_action = ACTION_RIGHT

    return int(horizontal_action)

  def get_player_action(self,horizontal_action):

    if type(self.ai) != type(None):
      return int(self.ai.response)

    if horizontal_action != ACTION_STAND_STILL:
      return int(horizontal_action)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
      return ACTION_UP

    if keys[pygame.K_DOWN]:
      return ACTION_DOWN

    return ACTION_STAND_STILL

  def move_player_left(self):

    if self.current_lane > 0:
      self.current_lane -= 1
      self.car_position[0] = int(self.game_map['lanes'][self.current_lane]['car_x'])

  def move_player_right(self):

    maximum_lane = (len(self.game_map['lanes'])- 1)

    if self.current_lane < maximum_lane:
      self.current_lane += 1
      self.car_position[0] = int(self.game_map['lanes'][self.current_lane]['car_x'])

  def move_player_up(self):

    self.car_position[1] -= self.player_velocity

  def move_player_down(self):

    self.car_position[1] += self.player_velocity

  def keep_player_on_screen(self):

    minimum_y = 0
    maximum_y = (self.screen_height - self.car_height)

    if self.car_position[1] < minimum_y:
      self.car_position[1] = int(minimum_y)

    elif self.car_position[1] > maximum_y:
      self.car_position[1] = int(maximum_y)

  def update_player(self,action):

    if action == ACTION_LEFT:
      self.move_player_left()

    elif action == ACTION_RIGHT:
      self.move_player_right()

    elif action == ACTION_UP:
      self.move_player_up()

    elif action == ACTION_DOWN:
      self.move_player_down()

    self.keep_player_on_screen()

  def generate_obstacle(self):

    lane_index = random.randint(0,len(self.game_map['lanes']) - 1)
    lane = (self.game_map['lanes'][lane_index])
    obstacle_direction = (lane['direction'])
    obstacle_x = int(lane['car_x'])

    if obstacle_direction == 'down':
      obstacle_y = int(-self.car_height)

    else:
      obstacle_y = int(self.screen_height)

    obstacle_velocity = (random.randint(self.min_obstacle_velocity, self.max_obstacle_velocity))
    obstacle_image = (random.choice(self.obstacle_car_images))

    return {
      'x': int(obstacle_x),
      'y': int(obstacle_y),
      'direction': (obstacle_direction),
      'velocity': int(obstacle_velocity),
      'lane_index': int(lane_index),
      'lane_name': (lane['name']),
      'image': (obstacle_image)
      }

  def try_generate_obstacle(self):

    random_number = (random.randint(0,self.obstacle_spawn_range))

    if (random_number < self.obstacle_spawn_threshold):
      obstacle = (self.generate_obstacle())
      self.obstacles.append(obstacle)

  def move_obstacle(self,obstacle):

    if (obstacle['direction'] == 'down'):
      obstacle['y'] += (obstacle['velocity'])

    elif (obstacle['direction'] == 'up'):
      obstacle['y'] -= (obstacle['velocity'])

    obstacle['y'] = int(obstacle['y'])

  def remove_passed_obstacles(self):

    active_obstacles = []

    for obstacle in self.obstacles:

      above_screen = (obstacle['y'] + self.car_height <= 0)
      below_screen = (obstacle['y'] >= self.screen_height)

      if above_screen or below_screen:
        continue

      active_obstacles.append(obstacle)

    self.obstacles = (active_obstacles)

  def update_obstacles(self):

    self.try_generate_obstacle()

    for obstacle in self.obstacles:
      self.move_obstacle(obstacle)

    self.remove_passed_obstacles()

  def get_player_rect(self):

    return pygame.Rect(
      int(self.car_position[0]),
      int(self.car_position[1]),
      int(self.car_width),
      int(self.car_height))

  def get_obstacle_rect(self,obstacle):

    return pygame.Rect(
      int(obstacle['x']),
      int(obstacle['y']),
      int(self.car_width),
      int(self.car_height))

  def check_collision(self):

    player_rect = (self.get_player_rect())

    for obstacle in self.obstacles:

      obstacle_rect = (self.get_obstacle_rect(obstacle))

      if player_rect.colliderect(obstacle_rect):
        return obstacle

    return None

  def handle_collision(self,obstacle):

    self.collided = True
    self.update_survival_time()

    print()
    print("COLLISION")
    print("Collision lane:",obstacle['lane_name'])
    print(
      f"Survival time: "
      f"{self.survival_time:.2f} seconds")

    if type(self.ai) != type(None):
      self.ai.initialize_game()
    self.running = False

  def get_direction_value(self,direction):

    if direction == 'down':
      return DIRECTION_DOWN

    elif direction == 'up':
      return DIRECTION_UP

    return DIRECTION_DOWN

  def get_obstacle_distance(self,obstacle):

    player_x = int(self.car_position[0])
    player_y = int(self.car_position[1])
    obstacle_x = int(obstacle['x'])
    obstacle_y = int(obstacle['y'])
    x_distance = abs(obstacle_x - player_x)
    y_distance = abs(obstacle_y - player_y)
    distance = (x_distance + y_distance)

    return int(distance)

  def get_observed_obstacles(self):

    sorted_obstacles = sorted(self.obstacles,key=self.get_obstacle_distance)
    observed_obstacles = (sorted_obstacles[:MAX_OBSERVED_OBSTACLES])

    return observed_obstacles

  def add_empty_obstacle_data(self,training_example):

    training_example.append(0)
    training_example.append(0)
    training_example.append(0)
    training_example.append(0)
    training_example.append(0)
    training_example.append(0)

  def collect_data(self,action):

    if action not in (0,1,2,3,4):
      raise ValueError(f"Invalid action collected: {action}")

    training_example = []

    training_example.append(int(self.car_position[0]))
    training_example.append(int(self.car_position[1]))
    training_example.append(int(self.current_lane))

    observed_obstacles = (self.get_observed_obstacles())

    for obstacle in observed_obstacles:

      training_example.append(1)
      training_example.append(int(obstacle['x']))
      training_example.append(int(obstacle['y']))
      training_example.append(int(obstacle['velocity']))
      training_example.append(int(self.get_direction_value(obstacle['direction'])))
      training_example.append(int(obstacle['lane_index']))

    missing_obstacles = (MAX_OBSERVED_OBSTACLES - len(observed_obstacles))

    for _ in range(missing_obstacles):
      self.add_empty_obstacle_data(training_example)

    expected_input_size = (3 + (MAX_OBSERVED_OBSTACLES * 6))

    if len(training_example) != expected_input_size:
      raise ValueError(
        f"Expected "
        f"{expected_input_size} "
        f"input values, got "
        f"{len(training_example)}")

    training_example.append(int(action))

    expected_sample_size = (expected_input_size + 1)

    if len(training_example) != expected_sample_size:
      raise ValueError(
        f"Expected "
        f"{expected_sample_size} "
        f"sample values, got "
        f"{len(training_example)}")

    self.data.append(training_example)

  def update_survival_time(self):

    current_time = (pygame.time.get_ticks())
    elapsed_milliseconds = (current_time - self.start_time)
    self.survival_time = (elapsed_milliseconds / 1000.0)

  def draw_survival_time(self):

    time_text = (
      f"Time: "
      f"{self.survival_time:.2f}s")

    text_surface = (self.font.render(time_text,True,(255,255,255)))
    text_rect = (text_surface.get_rect(top=10,right=(self.screen_width - 10)))

    self.screen.blit(text_surface,text_rect)

  def save_training_data(self,filename=train_data):
    number_of_samples = (len(self.data))
    total_integers = (number_of_samples * 34)

    if number_of_samples == 0:
      return

    all_prev = open(filename,'r').read().splitlines()

    with open(filename,'w') as file:
      for value in all_prev:
        file.write(str(value))
        file.write("\n")

      if self.survival_time > 60.00:
        for training_example in self.data[:len(self.data)-180]:
          for value in training_example:
            file.write(str(value))
            file.write("\n")

        file.write("/")
        file.write("\n")
      else:
        print("not saving this game's data")

    print()
    print(
      f"Training data saved to "
      f"{filename}")

    print(
      f"Collected samples: "
      f"{number_of_samples}")

    print(
      f"Total integers: "
      f"{total_integers}")

    print(
      f"Final survival time: "
      f"{self.survival_time:.2f} seconds")

  def update_game_state(self):

    if type(self.ai) != type(None):
      self.ai.response = self.ai_response()

    horizontal_action = (self.handle_events())

    if not self.running:
      return

    action = (self.get_player_action(horizontal_action))

    self.update_player(action)
    self.update_obstacles()

    collided_obstacle = (self.check_collision())

    if collided_obstacle is not None:
      self.handle_collision(collided_obstacle)
      return

    self.update_survival_time()
    self.collect_data(action)

  def draw_road(self):

    pygame.draw.rect(
      self.screen,
      (50,50,50),
      (self.game_map['road_left'],0,
      (self.game_map['road_right'] - self.game_map['road_left']),
      self.hwroad_height))
    
    pygame.draw.line(
      self.screen,
      (255,255,255),
      (self.game_map['left_outer_line'],0),
      (self.game_map['left_outer_line'],self.hwroad_height),5)

    pygame.draw.line(
      self.screen,
      (75,75,75),
      (self.game_map['first_inner_line'],0),
      (self.game_map['first_inner_line'],self.hwroad_height),5)

    pygame.draw.line(
      self.screen,
      (255,255,0),
      (self.game_map['median_left'],0),
      (self.game_map['median_left'],self.hwroad_height),7)

    pygame.draw.line(
      self.screen,
      (255,255,0),
      (self.game_map['median_right'],0),
      (self.game_map['median_right'],self.hwroad_height),7)

    pygame.draw.line(
      self.screen,
      (75,75,75),
      (self.game_map['second_inner_line'],0),
      (self.game_map['second_inner_line'],self.hwroad_height),5)

    pygame.draw.line(
      self.screen,
      (255,255,255),
      (self.game_map['right_outer_line'],0),
      (self.game_map['right_outer_line'],self.hwroad_height),5)

  def draw_obstacles(self):

    for obstacle in self.obstacles:
      self.screen.blit(
        obstacle['image'],
        (int(obstacle['x']),int(obstacle['y'])))

  def draw_player(self):

    self.screen.blit(
      self.player_car_image,
      (int(self.car_position[0]),int(self.car_position[1])))

  def render(self):

    self.screen.fill((0,0,0))
    self.draw_road()
    self.draw_obstacles()
    self.draw_player()
    self.draw_survival_time()

    pygame.display.flip()

  def run_game_loop(self):

    while self.running:

      self.update_game_state()

      if self.running:
        self.render()

      self.clock.tick(self.fps)

    if not self.collided:
      self.update_survival_time()

    if type(self.ai) == type(None):
      self.save_training_data(train_data)

    pygame.quit()