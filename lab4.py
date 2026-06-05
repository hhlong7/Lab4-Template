import stat
import dataclasses
from collections import deque
import os
from agents import EntityAgent, UncertainAgent
from model import Location,GameState, GameAction, WizardMoves, Location, Crystal, Portal, Lava, Wall, GameTransitions, Observation, EmptyTile, LocationCounts, LocationDistribution
import random

random.seed(int(os.environ.get("LAB4_SEED", "1")))


class MDP:
    def __init__(self,initial_state: GameState, escape_reward:float,living_reward: float,death_reward:float, discount: float):
        self.game_state = initial_state
        self.living_reward = living_reward
        self.death_reward = death_reward
        self.escape_reward = escape_reward
        self.discount = discount
        self._transition_cache: dict[tuple[int, int, GameAction], LocationDistribution] = {}

    def reward(self,source:GameState,target:GameState, action: GameAction) -> float:
        loc = target.active_entity_location
        if isinstance(target.tile_grid[loc.row][loc.col],Lava):
            return self.death_reward
        elif target.victory:
            return self.escape_reward
        else:
            return self.living_reward


    def transition_model(self,location: Location, action: GameAction) -> LocationDistribution:
        """
        Transition model of the MDP, gives conditional probability distribution of result location given starting location and action choice.
        """
        cache_key = (location.row, location.col, action)
        cached = self._transition_cache.get(cache_key)
        if cached is not None:
            return cached

        source_state = self.game_state.replace_active_entity_location(location)
        successors = GameTransitions.get_successors(source_state)
        actions = [a for a, _ in successors]
        successor_states = [state for _, state in successors]

        if action not in actions:
            result = self.transition_model(location,WizardMoves.STAY)
            self._transition_cache[cache_key] = result
            return result

        # Outcomes are either the desired outcome of the action, or a random other action each with 50% prob.
        possible_results = LocationCounts(self.game_state.grid_size)
        for i in range(len(actions)):
            if actions[i] == action:
                for _ in range(len(actions)+1):
                    possible_results.add_count(successor_states[i].active_entity_location)
            else:
                possible_results.add_count(successor_states[i].active_entity_location)

        result = possible_results.normalize()
        self._transition_cache[cache_key] = result
        return result

    def transition_distribution(self, source: LocationDistribution, action: GameAction) -> LocationDistribution:
        """
        Given a location distribution, calculate the new distribution that is a result of taking the given action.
        The easiest way to do this will involve sampling.
        """

        """
        make a new count grid, repeat for that many times,
        sample a location from source, get the action transition (transition model)
        count++, normalize at the end, return new distribution
        """

        rows, cols = self.game_state.grid_size
        next_prob_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for src_loc in source.locations():
            src_p = source.probability(src_loc)
            if src_p == 0.0:
                continue

            transition_dist = self.transition_model(src_loc, action)
            for target_loc in transition_dist.locations():
                trans_p = transition_dist.probability(target_loc)
                next_prob_grid[target_loc.row][target_loc.col] += src_p * trans_p

        return LocationDistribution(next_prob_grid)


class LocationValues:
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.value_grid = [[0.0 for _ in range(mdp.game_state.grid_size[1])] for _ in range(mdp.game_state.grid_size[0])]

    def value_iteration(self,k):
        for _ in range(k):
            self.value_iteration_update()

    def value_iteration_update(self):
        """
        Perform one update of value iteration based off of the provided MDP.
        """

        """
        make new grid, 
        loop through each locaiton, 
        make sure to skip walls or lava n portals,
        make a temp source state with the wizard at that loc, 
        use gametransition.get_successors for available actions,
        for each action, get transition distribution, 
        get q(s,a) by adding all possible next locs,
        use reward based on the next loc, 
        add discounted value of next, 
        set next value to max q(s,a)
        assign grid to next grid, 
        return that
        """

        next_value_grid = [[0.0 for _ in range(self.mdp.game_state.grid_size[1])] for _ in range(self.mdp.game_state.grid_size[0])]

        for r in range(self.mdp.game_state.grid_size[0]):
            for c in range(self.mdp.game_state.grid_size[1]):
                loc = Location(r, c)
                tile = self.mdp.game_state.tile_grid[r][c]

                """ skippin walls, lava n portal"""
                if isinstance(tile, (Wall, Lava, Portal)):
                    next_value_grid[r][c] = 0.0
                    continue

                source_state = self.mdp.game_state.replace_active_entity_location(loc)
                successors = GameTransitions.get_successors(source_state)
                if len(successors) == 0:
                    next_value_grid[r][c] = 0.0
                    continue

                best_action_value = float("-inf")
                for action, _ in successors:
                    transition_dist = self.mdp.transition_model(loc, action)
                    q_value = 0.0

                    for next_loc in transition_dist.locations():
                        p = transition_dist.probability(next_loc)
                        next_tile = self.mdp.game_state.tile_grid[next_loc.row][next_loc.col]

                        if isinstance(next_tile, Lava):
                            reward = self.mdp.death_reward
                        elif isinstance(next_tile, Portal):
                            reward = self.mdp.escape_reward
                        else:
                            reward = self.mdp.living_reward

                        q_value += p * (
                            reward + self.mdp.discount * self.value_grid[next_loc.row][next_loc.col]
                        )

                    if q_value > best_action_value:
                        best_action_value = q_value

                next_value_grid[r][c] = best_action_value

        self.value_grid = next_value_grid

        return next_value_grid


class MDPAgent(UncertainAgent):
    values: LocationValues
    current_position_estimate: LocationDistribution
    current_score_estimate: int
    mdp: MDP

    def __init__(self, mdp: MDP, value_iteration_steps=100):
        self.mdp = mdp
        self.values = LocationValues(mdp)
        self.values.value_iteration(value_iteration_steps)
        self.current_position_estimate = LocationDistribution.from_game_state(mdp.game_state)
        self.lava_count = sum(
            1
            for row in self.mdp.game_state.tile_grid
            for tile in row
            if isinstance(tile, Lava)
        )
        self.map_is_cliff = self.lava_count >= 5
        self.portal_distance_map = self._build_portal_distance_map()

    def _build_portal_distance_map(self) -> list[list[float]]:
        rows, cols = self.mdp.game_state.grid_size
        distances = [[float("inf") for _ in range(cols)] for _ in range(rows)]
        portal_loc = self.mdp.game_state.get_all_tile_locations(Portal)[0]
        queue = deque([portal_loc])
        distances[portal_loc.row][portal_loc.col] = 0.0

        while queue:
            current = queue.popleft()
            current_distance = distances[current.row][current.col]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                row = current.row + dr
                col = current.col + dc
                if not (0 <= row < rows and 0 <= col < cols):
                    continue
                if isinstance(self.mdp.game_state.tile_grid[row][col], (Wall, Lava)):
                    continue
                if distances[row][col] != float("inf"):
                    continue
                distances[row][col] = current_distance + 1.0
                queue.append(Location(row, col))

        return distances


    def observation_likelihood(self, observation: Observation, loc: Location)-> float:
        portal_loc = self.mdp.game_state.get_all_tile_locations(Portal)[0]
        portal_dist = abs(loc.row-portal_loc.row) + abs(loc.col - portal_loc.col)

        if abs(portal_dist - observation.approximatePortalDist) > 1:
            return 0
        else:
            return 1.0/3.0

    def update_prior(self,action: GameAction):
        self.current_position_estimate = self.mdp.transition_distribution(self.current_position_estimate,action)


    def update_belief(self, observation: Observation):
        """
        Use Bayes rule to update your beliefs about the wizard location by updating self.current_position_estimate.
        You have prior belief of your current estimate P(Loc), and the observation likelihood model (P(Obs | Loc)).
        Use these to calculate the new belief.
        """

        #We need to update our belief for all possible locations. So lets start by creating a new distribution
        new_estimate = LocationDistribution.from_game_state_uniform(self.mdp.game_state)

        """
        make new distribution container, loop thru them locs,
        get prev from current esitimate, get likelihood from observation likelihood,
        mult to get unnormalizred next, store it in new estimate
        renorm so it adds up to 1, save it to current estimate
        """

        total = 0.0
        for loc in new_estimate.locations():

            prev = self.current_position_estimate.probability(loc)
            likelihood = self.observation_likelihood(observation, loc)
            unnom_next = prev * likelihood
            new_estimate.update_probability(loc, unnom_next)
            total += unnom_next

        # If no location is compatible with the observation under current support, keep prior.
        if total == 0.0:
            return

        new_estimate.renormalize()
        self.current_position_estimate = new_estimate

    def react(self, observation: Observation) -> GameAction:
        """
        Our uncertain agent only has noisy observations to guess where in the dungeon it is. Use the previously implemented parts to generate an estimate for the distribution of possible locations the wizard might be at, updating every turn with a new observation, and choose the action based off of your value iteration policy based on your estimate.
        """

        # Use Bayes Rule to update your beliefs about where you think the wizard is based off of the observation
        self.update_belief(observation)


        # Part 2: Choose the best action
        # use your calculated value iteration map of location values alongside your estimated location to choose the best action given your uncertain state.
        # There are multiple ways to do this, but some things to consider:
            # 1. You want to select the action which will have the highest expected value, given the probability distribution of the resultant states.
            # 2. The expected value of some quantity is just the weighted average value of that quantity in an outcome weighted by the probability of that outcome over all outcomes (and can be estimated by the average of a sufficiently big sample of outcomes sampled from the distribution)
            # 3. You can sample locations from any distribution, and can form distributions of locations from samples
            # 4. You can find the distribution of the results of an action for a given specific location
            # 5. You can calculate the reward of a specific transition as a result of a specific action with a specific result
            # 6. You have an estimate of the value of each result location

        """
        update the belief from the observations
        either predict final loc distribution from action
        or compute expected value of that distribution
        pick the action with max expected value, return that
        """
        action = WizardMoves.STAY
        best_score = float("-inf")
        portal_loc = self.mdp.game_state.get_all_tile_locations(Portal)[0]
        lava_penalty = abs(self.mdp.death_reward) * (0.8 if self.map_is_cliff else 0.1)
        hazard_penalty = 2.0 if self.map_is_cliff else 0.4
        path_weight = 0.8 if self.map_is_cliff else 0.25

        def adjacent_lava_tiles(loc: Location) -> int:
            count = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                row = loc.row + dr
                col = loc.col + dc
                if 0 <= row < self.mdp.game_state.grid_size[0] and 0 <= col < self.mdp.game_state.grid_size[1]:
                    if isinstance(self.mdp.game_state.tile_grid[row][col], Lava):
                        count += 1
            return count

        for possible_action in WizardMoves:
            score = 0.0
            lava_probability = 0.0

            for current_loc in self.current_position_estimate.locations():
                belief_weight = self.current_position_estimate.probability(current_loc)
                if belief_weight == 0.0:
                    continue

                transition_dist = self.mdp.transition_model(current_loc, possible_action)
                for next_loc in transition_dist.locations():
                    transition_weight = transition_dist.probability(next_loc)
                    tile = self.mdp.game_state.tile_grid[next_loc.row][next_loc.col]

                    if isinstance(tile, Lava):
                        reward = self.mdp.death_reward
                        lava_probability += belief_weight * transition_weight
                    elif isinstance(tile, Portal):
                        reward = self.mdp.escape_reward
                    else:
                        reward = self.mdp.living_reward

                    portal_distance = self.portal_distance_map[next_loc.row][next_loc.col]
                    if portal_distance == float("inf"):
                        portal_distance = self.mdp.game_state.grid_size[0] * self.mdp.game_state.grid_size[1]
                    score += belief_weight * transition_weight * (
                        reward
                        + self.mdp.discount * self.values.value_grid[next_loc.row][next_loc.col]
                        - path_weight * portal_distance
                        - hazard_penalty * adjacent_lava_tiles(next_loc)
                    )

            score -= lava_penalty * lava_probability

            if score > best_score:
                best_score = score
                action = possible_action

        #When choosing an action, we must update our prior to account for the new distribution as a result of the action being taken
        self.update_prior(action)
        return action
