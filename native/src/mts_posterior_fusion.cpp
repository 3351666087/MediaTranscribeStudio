#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" int mts_viterbi_decode_labels(
    const float* emissions,
    const float* boundary_scores,
    int num_frames,
    int num_states,
    float switch_penalty,
    float stay_bonus,
    float boundary_relief,
    int* out_labels
) {
    if (emissions == nullptr || out_labels == nullptr || num_frames <= 0 || num_states <= 0) {
        return 1;
    }

    std::vector<float> dp(static_cast<std::size_t>(num_frames) * static_cast<std::size_t>(num_states), 0.0f);
    std::vector<int> back(static_cast<std::size_t>(num_frames) * static_cast<std::size_t>(num_states), 0);

    for (int state = 0; state < num_states; ++state) {
        dp[static_cast<std::size_t>(state)] = emissions[state];
    }

    for (int frame = 1; frame < num_frames; ++frame) {
        const float boundary = boundary_scores != nullptr ? boundary_scores[frame] : 0.0f;
        const float effective_switch = std::max(0.0f, switch_penalty - boundary_relief * boundary);

        const float* prev_row = dp.data() + static_cast<std::size_t>(frame - 1) * static_cast<std::size_t>(num_states);
        float best_prev_score = prev_row[0];
        int best_prev_state = 0;
        for (int state = 1; state < num_states; ++state) {
            if (prev_row[state] > best_prev_score) {
                best_prev_score = prev_row[state];
                best_prev_state = state;
            }
        }

        for (int state = 0; state < num_states; ++state) {
            const std::size_t offset = static_cast<std::size_t>(frame) * static_cast<std::size_t>(num_states) + static_cast<std::size_t>(state);
            const float stay_score = prev_row[state] + stay_bonus;
            const float switch_score = best_prev_score - effective_switch;
            if (stay_score >= switch_score) {
                dp[offset] = emissions[offset] + stay_score;
                back[offset] = state;
            } else {
                dp[offset] = emissions[offset] + switch_score;
                back[offset] = best_prev_state;
            }
        }
    }

    const float* last_row = dp.data() + static_cast<std::size_t>(num_frames - 1) * static_cast<std::size_t>(num_states);
    int best_state = 0;
    float best_score = last_row[0];
    for (int state = 1; state < num_states; ++state) {
        if (last_row[state] > best_score) {
            best_score = last_row[state];
            best_state = state;
        }
    }

    out_labels[num_frames - 1] = best_state;
    for (int frame = num_frames - 1; frame > 0; --frame) {
        const std::size_t offset = static_cast<std::size_t>(frame) * static_cast<std::size_t>(num_states)
            + static_cast<std::size_t>(out_labels[frame]);
        out_labels[frame - 1] = back[offset];
    }
    return 0;
}
