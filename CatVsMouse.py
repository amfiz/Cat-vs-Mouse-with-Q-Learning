import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time

class GridWorld:
    def __init__(self, grid_size=6):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.goal_state = (grid_size - 1, grid_size - 1)
        self.obstacles = [(2, 1), (3, 3), (4, 4),(2,1),(3,1)] 
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1

        if (x, y) in self.obstacles:
            reward = -10 
            done = True    
            self.state = (x, y)
            return self.state, reward, done
        
        self.state = (x, y)
        
        if self.state == self.goal_state:
            reward = 10 
            done = True   
        else:
            reward = -0.1 
            done = False
            
        return self.state, reward, done

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x, y = self.state
        gx, gy = self.goal_state
        grid[x, y] = -1
        grid[gx, gy] = 1  
        for ox, oy in self.obstacles:
            grid[ox, oy] = 2  
        return grid

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            x, y = state
            return self.env.actions[np.argmax(self.q_table[x, y])]

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.env.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.gamma * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action_idx]
        self.q_table[x, y, action_idx] += self.alpha * td_error

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class RLVisualizer(tk.Tk):
    def __init__(self, agent, env):
        super().__init__()
        self.agent = agent
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = 60
        self.title("Cat and Mouse RL Visualizer")
        self.geometry(f"{self.grid_size * self.cell_size}x{self.grid_size * self.cell_size + 50}")
        
       
        self.cat_image = ImageTk.PhotoImage(Image.open("cat.jpg").resize((self.cell_size - 10, self.cell_size - 10)))
        self.mouse_image = ImageTk.PhotoImage(Image.open("mouse.jpg").resize((self.cell_size - 10, self.cell_size - 10)))
        self.sad_mouse_image = ImageTk.PhotoImage(Image.open("sad mouse.png").resize((self.cell_size - 10, self.cell_size - 10)))
        self.sad_cat_image = ImageTk.PhotoImage(Image.open("sad cat.png").resize((self.cell_size - 10, self.cell_size - 10)))
        self.dog_image = ImageTk.PhotoImage(Image.open("mad dog.jpg").resize((self.cell_size - 10, self.cell_size - 10)))
        self.obstacle_image = self.dog_image 
        
        self.canvas = tk.Canvas(self, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size, bg='white')
        self.canvas.pack()
        self.reward_label = tk.Label(self, text="Current Reward: 0")
        self.reward_label.pack()
        self.cumulative_reward_label = tk.Label(self, text="Cumulative Reward: 0")
        self.cumulative_reward_label.pack()
        self.cumulative_reward = 0
        self.message_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.message_label.pack()
        self.visited_cells = []  
        self.after(0, self.update_grid)

    def render_grid(self, final_state=None):
        self.canvas.delete("all")
        grid = self.env.render()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                color = "white"
                if (i, j) in self.visited_cells:
                    color = "green" 
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill=color)
                if grid[i, j] == -1:
                    self.canvas.create_image(x0 + 5, y0 + 5, anchor='nw', image=self.cat_image) 
                elif grid[i, j] == 1:
                    if final_state == "cat_wins":
                        self.canvas.create_image(x0 + 5, y0 + 5, anchor='nw', image=self.cat_image)  
                    elif final_state == "mouse_eaten":
                        self.canvas.create_image(x0 + 5, y0 + 5, anchor='nw', image=self.sad_mouse_image)  
                    else:
                        self.canvas.create_image(x0 + 5, y0 + 5, anchor='nw', image=self.mouse_image) 
                elif grid[i, j] == 2:
                    self.canvas.create_image(x0 + 5, y0 + 5, anchor='nw', image=self.obstacle_image)  

    def update_grid(self):
        state = self.env.reset()
        self.visited_cells = [state]  
        done = False
        self.cumulative_reward = 0
        while not done:
            self.render_grid()
            self.update_idletasks()
            self.update()
            time.sleep(1)  
            action = self.agent.choose_action(state)
            next_state, reward, done = self.env.step(action)
            self.cumulative_reward += reward
            self.reward_label.config(text=f"Current Reward: {reward}")
            self.cumulative_reward_label.config(text=f"Cumulative Reward: {self.cumulative_reward}")
            state = next_state
            self.visited_cells.append(state) 
            if done:
                if reward == -10: 
                    self.render_grid()
                    self.canvas.create_image(state[1] * self.cell_size + 5, state[0] * self.cell_size + 5, anchor='nw', image=self.sad_cat_image)
                    self.message_label.config(text="Game Over - Dog ate the cat!")
                elif state == self.env.goal_state:
                    self.render_grid("mouse_eaten")
                    self.message_label.config(text="Cat is eating the mouse!")
                self.update_idletasks()
                self.update()
                time.sleep(2)
                if state == self.env.goal_state:
                    self.render_grid("cat_wins")
                    self.message_label.config(text="Cat has eaten the mouse successfully!")
                break
        self.render_grid("cat_wins" if state == self.env.goal_state else None)

if __name__ == "__main__":
    
    env = GridWorld(grid_size=6)
    agent = QLearningAgent(env)
    agent.train(num_episodes=500)
    
    app = RLVisualizer(agent, env)
    app.mainloop()
