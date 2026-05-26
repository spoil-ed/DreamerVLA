image_t, state_t --> state encoder ----> h_t
text -----------> text encoder -------> c

VLA head:
(h_t, c) -----------------------------> action_pred

WM head:
(h_t, action_t, c) -------------------> next_h_pred
