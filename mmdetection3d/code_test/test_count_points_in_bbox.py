import torch
import numpy as np
import plotly.graph_objects as go
from mmcv.ops.points_in_boxes import points_in_boxes_all


def count_points_in_bbox(points_list, bboxes_list):
    assert len(points_list) == len(bboxes_list), "points_list and bboxes_list must have the same length"
    results = []

    for points, bboxes in zip(points_list, bboxes_list):
        bboxes_bottom_center = bboxes.clone()
        bboxes_bottom_center[:, 2] -= bboxes_bottom_center[:, 5] / 2

        batched_points = points.unsqueeze(0)
        batched_boxes = bboxes_bottom_center.unsqueeze(0)

        mask = points_in_boxes_all(batched_points, batched_boxes)

        count = mask[0].sum(dim=0)
        results.append(count.to(dtype=torch.int32))

    return results


def draw_3d_box(fig, box, color='red', name='box'):
    x_center, y_center, z_center, x_size, y_size, z_size, rz = box.tolist()
    half_x = x_size / 2
    half_y = y_size / 2
    half_z = z_size / 2

    vertices = np.array([
        [-half_x, -half_y, -half_z],
        [ half_x, -half_y, -half_z],
        [ half_x,  half_y, -half_z],
        [-half_x,  half_y, -half_z],
        [-half_x, -half_y,  half_z],
        [ half_x, -half_y,  half_z],
        [ half_x,  half_y,  half_z],
        [-half_x,  half_y,  half_z],
    ])

    # 绕 z 轴旋转
    cos_rz = np.cos(rz)
    sin_rz = np.sin(rz)
    rot_z = np.array([[cos_rz, -sin_rz, 0],
                      [sin_rz, cos_rz,  0],
                      [0,       0,      1]])

    vertices = (rot_z @ vertices.T).T
    vertices += [x_center, y_center, z_center]

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    for edge in edges:
        start, end = edge
        fig.add_trace(go.Scatter3d(
            x=[vertices[start, 0], vertices[end, 0]],
            y=[vertices[start, 1], vertices[end, 1]],
            z=[vertices[start, 2], vertices[end, 2]],
            mode='lines',
            line=dict(color=color, width=5),
            name=name,
            showlegend=(edge == [0,1])
        ))


def visualize_sample(points, bboxes, box_assignments=None, title="Sample Visualization"):
    points_np = points.cpu().numpy()
    bboxes_np = bboxes.cpu().numpy()

    if box_assignments is not None:
        box_assignments = box_assignments.cpu().numpy()

    fig = go.Figure()

    if box_assignments is not None and box_assignments.size > 0:
        unique_box_ids = np.unique(box_assignments)
        colors = {}
        for idx, box_id in enumerate(unique_box_ids):
            colors[box_id] = f'rgb({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)})'

        for i, point in enumerate(points_np):
            box_id = box_assignments[i]
            color = colors.get(box_id, 'gray') if box_id != -1 else 'gray'
            fig.add_trace(go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],
                mode='markers',
                marker=dict(size=5, color=color),
                showlegend=False
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Points'
        ))

    for i, box in enumerate(bboxes_np):
        draw_3d_box(fig, box, color=f'hsl({i * 40}, 100%, 50%)', name=f"Box {i}")

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=title)
    output_filename = f"{title.replace(' ', '_')}.html"
    fig.write_html(output_filename)
    print(f"Visualization saved as {output_filename}")


def test_count_points_in_bbox():
    print("Start test_count_points_in_bbox...")

    points_list = [
        torch.tensor([[0.5, 0.5, 0.5], [1.4, 1.4, 1.4], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0]]),
        torch.tensor([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [2.9, 2.9, 2.9]]),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.2, 0.0], [1.0, 0.3, 0.0]])
    ]

    bboxes_list = [
        torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
        torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]]),
        torch.tensor([[1.0, 0.2, 0.0, 0.5, 0.5, 0.5, np.pi / 4]])  # 旋转 π/4 的边界框
    ]

    points_list = [points.to("cuda") for points in points_list]
    bboxes_list = [bboxes.to("cuda") for bboxes in bboxes_list]

    results = count_points_in_bbox(points_list, bboxes_list)

    for i, res in enumerate(results):
        print(f"Sample {i+1}: {res.tolist()}")

    batched_points = torch.nn.utils.rnn.pad_sequence(points_list, batch_first=True)
    batched_boxes = torch.nn.utils.rnn.pad_sequence(bboxes_list, batch_first=True)
    mask = points_in_boxes_all(batched_points, batched_boxes)

    box_assignments = []
    for b in range(len(points_list)):
        valid_mask = mask[b, :, :bboxes_list[b].shape[0]]
        assigned = valid_mask.argmax(dim=1)
        background_mask = valid_mask.sum(dim=1) == 0
        assigned[background_mask] = -1
        box_assignments.append(assigned)

    for i in range(len(points_list)):
        visualize_sample(points_list[i], bboxes_list[i], box_assignments[i], title=f"Sample {i+1} Visualization")

    expected = [
        torch.tensor([2, 1]),
        torch.tensor([2, 1, 0]),
        torch.tensor([4])  # 所有点都在旋转框内
    ]
    expected = [res.to(torch.int32) for res in expected]

    for i, (actual, expect) in enumerate(zip(results, expected)):
        assert torch.equal(actual.cpu(), expect), f"Test failed at sample {i+1}, got {actual}, expected {expect}"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_count_points_in_bbox()