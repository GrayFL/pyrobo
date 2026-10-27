"""Vectorized objective terms and hard intent masks; candidate rows stay aligned."""
import numpy as np

from ...core import MotionPolicy
from pyrobo.core.geometry import wrap_angle
from .path import polyline_distances


def navigation_scores(context, search, poses, config, space, observation):
    final = poses
    speed, turn = search[:, 0], search[:, 1]
    c = context
    if c.mode == 'aligning':
        heading = np.abs(wrap_angle(final[:, 2] - c.goal[2]))
        score = 4*heading + 3*np.linalg.norm(final[:, :2] - c.goal[:2], axis=1) + abs(speed) + .15*abs(turn)
    elif c.mode == 'turn_braking':
        score = 5*turn**2 + 3*abs(speed)
    elif c.mode == 'turning':
        error = c.signed_error
        if error*c.turn_direction < 0:
            error += c.turn_direction*2*np.pi
        heading = np.abs(final[:, 2] - c.state[2] - error)
        score = 4*heading + 3*abs(speed) + .1*((turn - c.desired[1])/space.bounds()[1][1])**2
    else:
        distance = np.linalg.norm(final[:, :2] - c.target, axis=1)
        path_distance = polyline_distances(final[:, :2], c.remaining)
        heading = 1 - np.cos(final[:, 2] - c.direction)
        max_speed = space.bounds()[1][0]
        score = (config.position_weight*distance + config.path_weight*path_distance
                 + config.heading_weight*heading + config.speed_weight*((speed-c.desired[0])/max_speed)**2)
        score += .15*np.maximum(-speed, 0.) / max(-space.bounds()[0][0], .01)
        if not space.can_spin and c.remaining_distance < config.lookahead:
            weight = 2*(1 - min(c.distance_to_goal/config.lookahead, 1))
            score += weight*(1 - np.cos(final[:, 2] - c.goal[2]))
            if c.distance_to_goal <= config.goal_tolerance:
                score += 2*np.linalg.norm(final[:, :2] - c.goal[:2], axis=1)
        if space.can_spin and abs(observation.chassis.velocity.angular) > 1e-3:
            reversal = np.maximum(-turn*np.sign(observation.chassis.velocity.angular), 0)
            score += config.yaw_reversal_weight*(reversal/max(config.angular_acceleration*observation.dt, .1))**2
    return score, np.ones(len(search), dtype=bool)


def teleop_scores(context, search, poses, config, space):
    desired = context.desired
    scale = np.maximum(space.bounds()[1], .1)
    delta = (search - desired) / scale
    score = 4*delta[:, 0]**2 + .2*delta[:, 1]**2
    if context.previous is not None:
        score += .06*np.sum(((search-context.previous)/scale)**2, axis=1)
    valid = np.ones(len(search), dtype=bool)
    if context.request.policy == MotionPolicy.TELEOP_AVOID:
        yaw = context.state[2]
        offset = poses[:, :2] - context.state[:2]
        lateral = -np.sin(yaw)*offset[:, 0] + np.cos(yaw)*offset[:, 1]
        actual_heading = wrap_angle(poses[:, 2] - yaw)
        requested_yaw = space.yaw_reference(desired[None])[0]
        # Bound deviation from the user's commanded arc, not from a made-up goal.
        angle = requested_yaw * config.prediction_time
        lateral_ref = desired[0] * config.prediction_time * np.sinc(angle/(2*np.pi)) * np.sin(angle/2)
        valid &= abs(lateral-lateral_ref) <= context.request.max_lateral_deviation + 1e-9
        valid &= abs(wrap_angle(actual_heading-angle)) <= context.request.max_heading_deviation + 1e-9
        score += .08*(lateral-lateral_ref)**2
        if context.preferred_turn_sign:
            reversal = np.maximum(-(search[:, 1]-desired[1])*context.preferred_turn_sign, 0.)
            score += .15*(reversal/scale[1])**2
    return score, valid
