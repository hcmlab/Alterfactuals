class SSIMLoss:

    def assert_gte_0(self, data):
        assert data.min() >= - 0.001  # error margin for 0

    def assert_le_1(self, data):
        assert data.max() <= 1.001  # error margin for 1
