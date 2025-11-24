from pathlib import Path
from unittest.mock import Mock, patch

from magnet_pinn.data.utils import worker_init_fn


def test_worker_init_fn_splits_simulations_evenly_with_two_workers():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations = [Path("/sim0"), Path("/sim2")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_splits_simulations_evenly_with_two_workers_second_worker():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 1

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(1)

    expected_simulations = [Path("/sim1"), Path("/sim3")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_splits_simulations_with_four_workers():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3"),
        Path("/sim4"),
        Path("/sim5"),
        Path("/sim6"),
        Path("/sim7")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 4
    mock_worker_info.id = 2

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(2)

    expected_simulations = [Path("/sim2"), Path("/sim6")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_handles_uneven_split():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3"),
        Path("/sim4")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 3
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations = [Path("/sim0"), Path("/sim3")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_handles_uneven_split_second_worker():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3"),
        Path("/sim4")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 3
    mock_worker_info.id = 1

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(1)

    expected_simulations = [Path("/sim1"), Path("/sim4")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_handles_uneven_split_third_worker():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3"),
        Path("/sim4")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 3
    mock_worker_info.id = 2

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(2)

    expected_simulations = [Path("/sim2")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_single_worker():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 1
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations = [Path("/sim0"), Path("/sim1"), Path("/sim2")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_with_single_simulation():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [Path("/sim0")]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations = [Path("/sim0")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_with_single_simulation_second_worker():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [Path("/sim0")]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 1

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(1)

    expected_simulations: list = []
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_with_empty_simulation_list():
    mock_dataset = Mock()
    mock_dataset.simulation_list = []

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations: list = []
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_with_string_paths():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        "/path/to/sim0",
        "/path/to/sim1",
        "/path/to/sim2",
        "/path/to/sim3"
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations = ["/path/to/sim0", "/path/to/sim2"]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_with_many_workers():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path(f"/sim{i}") for i in range(16)
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 8
    mock_worker_info.id = 3

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(3)

    expected_simulations = [Path("/sim3"), Path("/sim11")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_preserves_order():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/z_sim"),
        Path("/a_sim"),
        Path("/m_sim"),
        Path("/b_sim")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 0

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(0)

    expected_simulations = [Path("/z_sim"), Path("/m_sim")]
    assert mock_dataset.simulation_list == expected_simulations


def test_worker_init_fn_worker_id_parameter_ignored():
    mock_dataset = Mock()
    mock_dataset.simulation_list = [
        Path("/sim0"),
        Path("/sim1"),
        Path("/sim2"),
        Path("/sim3")
    ]

    mock_worker_info = Mock()
    mock_worker_info.dataset = mock_dataset
    mock_worker_info.num_workers = 2
    mock_worker_info.id = 1

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        worker_init_fn(999)

    expected_simulations = [Path("/sim1"), Path("/sim3")]
    assert mock_dataset.simulation_list == expected_simulations
