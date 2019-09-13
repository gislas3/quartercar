import quartercar.qc 

class MockProfile():
  def get_car_sample(velocities, distances, sample_rate_hz):
    return [1, 1, 1]

# def test_constant_velocity_and_distance():
#   model = qc.QC(0, 0, 0, 0, 0)
#   actual = model.run(MockProfile(), 1, 1)
#   expected = 