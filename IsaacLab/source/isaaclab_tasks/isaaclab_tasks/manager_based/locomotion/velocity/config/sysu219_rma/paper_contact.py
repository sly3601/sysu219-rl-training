"""Contact-force collection for the four zero-period RMA foot sensors."""

import torch
import warp as wp

from isaaclab.sensors import ContactSensor


@wp.kernel(enable_backward=False)
def _sum_friction(
    forces: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    sums: wp.array(dtype=wp.vec3),
):
    row = wp.tid()
    total = wp.vec3(0.0, 0.0, 0.0)
    for j in range(counts[row]):
        total += forces[starts[row] + j]
    sums[row] = total


class PaperFootContactSensor(ContactSensor):
    """A zero-period sensor expires for every environment on every physics step."""

    def _initialize_impl(self):
        super()._initialize_impl()
        self._per_step = self.cfg.update_period == 0 and self.cfg.history_length == 0 and not self.cfg.track_air_time
        self._all_pending = True
        self._refresh_pending = True

    def update(self, dt, force_recompute=False):
        if not self._per_step:
            return super().update(dt, force_recompute)
        self._all_pending = self._refresh_pending = True
        if force_recompute or self._is_visualizing:
            self._update_outdated_buffers()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._refresh_pending = True
        if env_ids is None:
            self._all_pending = True

    def _update_outdated_buffers(self):
        if not self._per_step:
            return super()._update_outdated_buffers()
        if self._refresh_pending:
            if self._all_pending:
                self._update_buffers_impl(range(self._num_envs))
                self._is_outdated.fill_(False)
            else:
                # Between physics steps a partial reset invalidates only its own rows.
                super()._update_outdated_buffers()
            self._all_pending = self._refresh_pending = False

    def _unpack_contact_buffer_data(
        self, contact_data, buffer_count, buffer_start_indices, avg=True, default=float("nan")
    ):
        if avg or default != 0.0 or not contact_data.is_cuda or contact_data.dtype != torch.float32:
            return super()._unpack_contact_buffer_data(contact_data, buffer_count, buffer_start_indices, avg, default)

        # PhysX supplies one start and count per body/filter pair. Reduce all its
        # contacts on-device without copying the total contact count to the CPU.
        counts = buffer_count.reshape(-1).to(torch.int32).contiguous()
        starts = buffer_start_indices.reshape(-1).to(torch.int32).contiguous()
        sums = contact_data.new_empty((counts.numel(), 3))
        wp.launch(
            _sum_friction,
            dim=counts.numel(),
            inputs=[
                wp.from_torch(contact_data.reshape(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(counts),
                wp.from_torch(starts),
            ],
            outputs=[wp.from_torch(sums, dtype=wp.vec3)],
            device=str(contact_data.device),
            stream=wp.stream_from_torch(contact_data.device),
        )
        return sums.view(self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3)
