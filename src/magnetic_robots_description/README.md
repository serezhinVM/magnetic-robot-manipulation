# magnetic_robots_description

Описание ячейки магнитного управления: рука **AUBO i5** с постоянным магнитом
на фланце, магнитный мягкий стержень (заглушка), точечные магнитные роботы и
фантом-сосуд для симуляции медицинского сценария.

## Структура

- `urdf/magnetic_cell.xacro` — рука `aubo_i5` (из `aubo_description`) + `magnet_link`
  на фланце `wrist3_Link` (standoff 60 мм, чтобы сталь кисти меньше искажала поле)
  + статическая сцена: стержень, 3 точечных робота, прозрачный сосуд.
- `launch/display.launch.py` — robot_state_publisher + joint_state_publisher_gui + RViz2.
- `rviz/magnetic_cell.rviz` — конфигурация RViz.

## Запуск

```bash
cd ~/ros2_ws
colcon build --packages-select aubo_description magnetic_robots_description
source install/setup.bash
ros2 launch magnetic_robots_description display.launch.py
```

В RViz появится рука AUBO i5 с красным магнитом на фланце; ползунки суставов
(joint_state_publisher_gui) двигают руку.

## Проверка TF магнита

```bash
ros2 run tf2_ros tf2_echo wrist3_Link magnet_link
```

Ожидается фиксированный сдвиг `z = 0.06` (standoff магнита).

## Параметры магнита

Задаются свойствами в `urdf/magnetic_cell.xacro`:

| Свойство | Значение | Описание |
|---|---|---|
| `magnet_radius` | 0.025 м | радиус цилиндрического магнита (D50 класс) |
| `magnet_length` | 0.025 м | длина магнита |
| `magnet_standoff` | 0.060 м | вынос магнита от фланца `wrist3_Link` |

## Следующие шаги (по плану)

- Шаг 2: `magnetic_field_model` — расчёт B(x), ∇B(x).
- Шаг 3: `continuum_rod_model` — стержень Коссера.
- Шаг 4: `continuum_sim_node` — замкнутая симуляция в RViz (MarkerArray).
