from gymnasium.envs.registration import register

register(
    id="HoloNav-v0", 
    entry_point="holonav.env:HoloNav",
	)