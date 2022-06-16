from vi import Agent, Config, Simulation

class MyAgent(Agent):

    def update(self):  # change these values to change agent behaviour
        if self.on_site():
            self.freeze_movement()

x = 375
y = 375

(
    Simulation(Config(radius=15))
    .spawn_site("images/green.png", x, y)  # spawn_site and chosen image
    .batch_spawn_agents(
        100,      # number of agents
        MyAgent,  # use our own MyAgent class
        images=["images/white.png",
                ],
    )
    .spawn_site("images/bigger_circle.png", x//4, y//2)
    .spawn_site("images/circle.png", (x//4)*3, y//2)
    .run()
)
