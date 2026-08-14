import math
import unittest

from geomem_env import (
    default_environments,
    egocentric_noisy_landmark_channel,
    effective_commutativity,
    environment_summary,
    gaussian_local_state_channel,
    gaussian_sensory_channel,
    local_view_observation,
    numeric_state_observation,
    observation_ambiguity,
    observe_behavior_state,
    observe_room_state,
    observe_trace_state,
    policy_ambiguity,
    registered_environment_specs,
    room_graph_environment,
    sampled_observation_ambiguity,
    sampled_residual_history_diagnostics,
    sensory_alias_key,
    sparse_observation_symbol,
    sparse_visual_feature_channel,
    standard_diagnostic_table,
    standard_observation_specs,
    standard_sensory_diagnostic_table,
    trace_monoid_general,
    trace_diagnostic_table,
    trace_monoid_family,
    transition_ambiguity,
    validate_environment,
)


class EnvironmentTests(unittest.TestCase):
    def test_default_environments_validate(self) -> None:
        envs = default_environments()
        self.assertGreaterEqual(len(envs), 10)
        for env in envs:
            self.assertEqual(validate_environment(env), [], env.name)

    def test_metadata_and_summary(self) -> None:
        env = room_graph_environment("bottleneck")
        summary = environment_summary(env)
        self.assertEqual(summary["family"], "room_graph")
        self.assertEqual(summary["geometry"], "bottleneck")
        self.assertEqual(summary["axes"]["rooms"], 6)
        self.assertGreater(summary["n_valid_transitions"], 0)

    def test_gymnasium_shaped_helpers(self) -> None:
        env = room_graph_environment("loop_rich")
        state, info = env.reset(seed=7, options={"start": (0, 1, 1)})
        self.assertEqual(info["seed"], 7)
        self.assertEqual(state, (0, 1, 1))
        self.assertIn("valid_actions", info)
        result = env.step_result(state, env.valid_actions(state)[0])
        self.assertIn(result.state, env.states)
        self.assertIn("valid_actions", result.info)

    def test_trace_diagnostics_recover_expected_trend(self) -> None:
        envs = trace_monoid_family(max_depth=5)
        scores = [
            effective_commutativity(env, length=3, n_sequences=50, seed=3, valid_only=True)
            for env in envs
        ]
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[-1], 1.0)
        self.assertLess(scores[1], scores[-1])

    def test_ambiguity_diagnostics_accept_observation_functions(self) -> None:
        env = trace_monoid_family(max_depth=5)[0]
        counts = lambda state: observe_trace_state(state, "counts", n_actions=len(env.actions))
        observation = observation_ambiguity(env, counts)
        transition = transition_ambiguity(env, counts)
        policy = policy_ambiguity(env, counts)
        self.assertGreater(observation["normalized_ambiguity"], 0.0)
        self.assertGreater(transition["weighted_ambiguous_fraction"], 0.0)
        self.assertGreater(policy["weighted_policy_ambiguity"], 0.0)

    def test_room_observations_compare_aliasing(self) -> None:
        env = room_graph_environment("tree_like")
        full = observation_ambiguity(env, lambda state: observe_room_state(state, "full"))
        local = observation_ambiguity(env, lambda state: observe_room_state(state, "local_position"))
        self.assertEqual(full["normalized_ambiguity"], 0.0)
        self.assertGreater(local["normalized_ambiguity"], 0.0)

    def test_behavioral_navigation_observations_compare_aliasing(self) -> None:
        env = default_environments()[10]
        self.assertEqual(env.name, "aliased_rooms")
        full = observation_ambiguity(env, lambda state: observe_behavior_state(state, env.name, "full"))
        local = observation_ambiguity(env, lambda state: observe_behavior_state(state, env.name, "local_symbol"))
        self.assertEqual(full["normalized_ambiguity"], 0.0)
        self.assertGreater(local["normalized_ambiguity"], 0.0)
        self.assertIsInstance(sparse_observation_symbol(env.states[0], env.name), int)

    def test_richer_observation_vectors_are_fixed_width(self) -> None:
        env = room_graph_environment("loop_rich")
        state = env.states[0]
        self.assertEqual(len(numeric_state_observation(state, width=6)), 6)
        self.assertEqual(len(local_view_observation(env, state)), len(env.actions) + 6)

    def test_trace_diagnostic_table_has_no_nan_for_standard_modes(self) -> None:
        rows = trace_diagnostic_table(trace_monoid_family(max_depth=4))
        self.assertEqual(len(rows), 16)
        for row in rows:
            for value in row.values():
                if isinstance(value, float):
                    self.assertFalse(math.isnan(value), row)

    def test_registry_specs_construct_environments(self) -> None:
        specs = registered_environment_specs()
        self.assertGreaterEqual(len(specs), 10)
        for spec in specs:
            env = spec.constructor()
            self.assertEqual(validate_environment(env), [], spec.name)

    def test_general_trace_constructor_supports_formal_sweep(self) -> None:
        env = trace_monoid_general(4, frozenset({("A", "B"), ("C", "D")}), 4, relation_id=2)
        self.assertEqual(env.metadata.family, "trace_monoid")
        self.assertEqual(env.metadata.axes[0][0], "alphabet_size")
        self.assertEqual(validate_environment(env), [])

    def test_standard_diagnostic_table_covers_taxonomy_and_expectations(self) -> None:
        envs = default_environments()
        rows = standard_diagnostic_table(envs)
        families = {env.metadata.family for env in envs}
        row_families = {str(row["family"]) for row in rows}
        self.assertTrue(families.issubset(row_families))
        self.assertGreaterEqual(len(rows), len(envs) * 2)
        for row in rows:
            self.assertIn("residual_history_score", row)
            self.assertIn("expect_vector_count", row)
            self.assertIn("expect_hybrid", row)

    def test_standard_observation_specs_are_nonempty(self) -> None:
        for env in default_environments():
            specs = standard_observation_specs(env)
            self.assertGreaterEqual(len(specs), 2, env.name)
            for spec in specs:
                self.assertIsNotNone(spec.observe(env.states[0]))

    def test_gaussian_local_state_channel_is_seeded_and_fixed_width(self) -> None:
        env = room_graph_environment("loop_rich")
        channel = gaussian_local_state_channel(env, sigma=0.1, width=5)
        state = env.states[3]
        self.assertEqual(channel.dim, 5)
        self.assertEqual(channel.observe(state, 11), channel.observe(state, 11))
        self.assertNotEqual(channel.observe(state, 11), channel.observe(state, 12))

    def test_gaussian_sensory_channel_aliases_without_action_bits(self) -> None:
        env = room_graph_environment("bottleneck")
        channel = gaussian_sensory_channel(env, sigma=0.0, dim=7, seed=13)
        first = (0, 1, 1)
        alias = (1, 1, 1)
        self.assertEqual(sensory_alias_key(env, first), sensory_alias_key(env, alias))
        self.assertEqual(channel.observe(first, 5), channel.observe(alias, 99))
        self.assertEqual(channel.dim, 7)
        self.assertNotEqual(channel.dim, len(env.actions))
        self.assertEqual(dict(channel.metadata)["includes_actions"], False)

    def test_gaussian_sensory_channel_noise_and_validation(self) -> None:
        env = room_graph_environment("loop_rich")
        channel = gaussian_sensory_channel(env, sigma=0.1, dim=5, seed=2)
        state = env.states[3]
        self.assertEqual(channel.observe(state, 11), channel.observe(state, 11))
        self.assertNotEqual(channel.observe(state, 11), channel.observe(state, 12))
        with self.assertRaises(ValueError):
            gaussian_sensory_channel(env, sigma=-0.1)
        with self.assertRaises(ValueError):
            gaussian_sensory_channel(env, dim=0)

    def test_sparse_visual_feature_channel_aliases_rooms(self) -> None:
        env = room_graph_environment("bottleneck")
        channel = sparse_visual_feature_channel(env, n_features=6, alias_rooms=True)
        by_obs: dict[tuple[float, ...], set[int]] = {}
        for state in env.states:
            by_obs.setdefault(channel.observe(state, 7), set()).add(state[0])
        self.assertTrue(any(len(rooms) > 1 for rooms in by_obs.values()))

    def test_egocentric_noisy_landmark_channel_width_and_seed(self) -> None:
        env = room_graph_environment("tree_like")
        channel = egocentric_noisy_landmark_channel(env, sigma=0.05, n_landmarks=7)
        state = env.states[0]
        self.assertEqual(channel.dim, len(env.actions) + 7)
        self.assertEqual(len(channel.observe(state, 5)), channel.dim)
        self.assertEqual(channel.observe(state, 5), channel.observe(state, 5))

    def test_sampled_channel_diagnostics_are_reproducible(self) -> None:
        env = room_graph_environment("tree_like")
        channel = sparse_visual_feature_channel(env, n_features=5, alias_rooms=True)
        first = sampled_observation_ambiguity(env, channel, n_samples=2, seed=3)
        second = sampled_observation_ambiguity(env, channel, n_samples=2, seed=3)
        residual = sampled_residual_history_diagnostics(env, channel, n_samples=2, seed=3)
        self.assertEqual(first, second)
        self.assertGreater(first["normalized_ambiguity"], 0.0)
        self.assertIn("residual_history_score", residual)

    def test_standard_sensory_diagnostic_table_uses_channels(self) -> None:
        rows = standard_sensory_diagnostic_table([room_graph_environment("tree_like")], sigma=0.0, dim=6)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["observation_mode"], "gaussian_sensory")
        self.assertIn("observation_n_observation_bins", row)
        self.assertIn("residual_history_score", row)
        self.assertIn("expect_hybrid", row)


if __name__ == "__main__":
    unittest.main()
