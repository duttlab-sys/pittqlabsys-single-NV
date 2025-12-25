import pytest
from unittest.mock import MagicMock, patch
import logging

from proteus_driver import ProteusDriver  # Import your ProteusDriver class


# Fixture to create the mock instrument and driver instance
@pytest.fixture
def mock_proteus_driver():
    with patch("proteus_driver.TepAdmin") as MockTepAdmin, patch("proteus_driver.TEVisaInst") as MockTEVisaInst:
        mock_admin = MockTepAdmin.return_value
        mock_instrument = MockTEVisaInst.return_value
        # Mock the send_scpi_query and send_scpi_cmd methods
        mock_instrument.send_scpi_query = MagicMock(return_value="Mocked Response")
        mock_instrument.send_scpi_cmd = MagicMock()
        mock_instrument.WriteBinaryData = MagicMock()
        mock_admin.open_instrument = MagicMock(return_value=mock_instrument)

        # Create an instance of the ProteusDriver with a mock instrument
        driver = ProteusDriver("192.168.0.100")
        driver.inst = mock_instrument  # Assign the mocked instrument
        yield driver


# Test the initialization and connection of ProteusDriver
def test_proteus_driver_initialization(mock_proteus_driver):
    driver = mock_proteus_driver
    driver._connect()  # Call the connect method, which should interact with the mock

    # Check that the instrument was initialized and the connection was established
    driver.inst.send_scpi_query.assert_called_with("*IDN?")
    assert driver.model_name is not None
    assert driver.paranoia_level == 2


# Test the send_command function
def test_send_command(mock_proteus_driver):
    driver = mock_proteus_driver
    response = driver.send_command("VOLT RANG?", query=True)

    # Ensure that send_scpi_query was called with the correct command
    driver.inst.send_scpi_query.assert_called_with("VOLT RANG?")
    assert response == "Mocked Response"


# Test the error handling in send_command (paranoia level 2)
def test_send_command_paranoia_error(mock_proteus_driver):
    driver = mock_proteus_driver
    # Simulate an error response
    driver.inst.send_scpi_query = MagicMock(return_value="Error: Some SCPI error")

    # Expect that an error is raised when the error queue is checked
    with pytest.raises(RuntimeError):
        driver.send_command("VOLT RANG?", query=True)


# Test the get_voltage_range function
def test_get_voltage_range(mock_proteus_driver):
    driver = mock_proteus_driver
    driver.send_command = MagicMock(return_value="0.0, 1.2, 0.5")

    result = driver.get_voltage_range()
    assert result == "0.0, 1.2, 0.5"  # Check that the command's response was processed correctly


# Test the reset function
def test_reset(mock_proteus_driver):
    driver = mock_proteus_driver
    driver.send_command = MagicMock()

    driver.reset()

    # Check that the reset commands were sent to the device
    driver.send_command.assert_any_call('*RST')
    driver.send_command.assert_any_call('*CLS')


# Test the configure_sampling_mode function
def test_configure_sampling_mode(mock_proteus_driver):
    driver = mock_proteus_driver

    # Mock the return values of methods used within configure_sampling_mode
    driver.get_granularity = MagicMock(return_value=16)
    driver.get_channels = MagicMock(return_value=([1, 2], [1, 2]))

    # Call the method with a sample target sampling rate
    target_sampling_rate = 2.5e9
    result = driver.configure_sampling_mode(target_sampling_rate)

    # Verify that the result matches the expected output
    assert result[0] == 16  # DAC Mode
    assert result[1] == 4   # Interpolation
    assert result[2] == 2.5e9  # Baseband rate


# Test if the correct exception is raised when an invalid voltage is set
def test_set_voltage_invalid(mock_proteus_driver):
    driver = mock_proteus_driver

    with pytest.raises(ValueError, match="Invalid voltage"):
        driver.set_voltage(1.5)  # Voltage above the allowed range


# Test the setup_sequence function (high-level test)
def test_setup_sequence(mock_proteus_driver):
    driver = mock_proteus_driver

    # Mock the methods used in setup_sequence
    driver.send_command = MagicMock()

    # Test setup sequence
    seqfilename = "mock_sequence.seq"
    driver.setup_sequence(seqfilename)

    # Check that relevant setup commands were sent
    driver.send_command.assert_any_call(':INST: CHAN 1')
    driver.send_command.assert_any_call(':INST: CHAN 2')
    driver.send_command.assert_any_call(f'SOUR1:FUNC:USER "{seqfilename}","MAIN"')
    driver.send_command.assert_any_call(f'SOUR2:FUNC:USER "{seqfilename}","MAIN"')


# Test logging behavior
def test_logging(mock_proteus_driver):
    with patch("proteus_driver.logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Trigger a log message by calling the method
        mock_proteus_driver._connect()

        # Check if logging.info was called
        mock_logger.info.assert_called_with("Proteus connection successful")


import pytest
from unittest.mock import MagicMock


@pytest.fixture
def instrument():
    # Assuming your class is called `InstrumentControl`
    instrument = InstrumentControl()
    instrument.send_command = MagicMock()  # Mocking the send_command method
    instrument.logger = MagicMock()  # Mock logger to avoid actual logging output
    return instrument


def test_set_ch1_marker2_laser_on(instrument):
    # Call the method
    instrument.set_ch1_marker2_laser_on()

    # Verify that the correct commands were sent
    instrument.send_command.assert_any_call(':INST:CHAN 1')  # Check if it selected channel 1
    instrument.send_command.assert_any_call(':MARK:SEL 2')  # Check if Marker 2 was selected
    instrument.send_command.assert_any_call(':MARK:VOLT:PTOP 0')  # Check if PTOP was set to 0
    instrument.send_command.assert_any_call(':MARK:VOLT:OFFS 2.0')  # Check if OFFS was set to 2.0
    instrument.send_command.assert_any_call(':MARK ON')  # Ensure the marker was turned on
    instrument.logger.info.assert_called_with("Turning on CH1 Marker 2 (laser control)")  # Check logging


def test_set_ch1_marker2_laser_off(instrument):
    # Call the method
    instrument.set_ch1_marker2_laser_off()

    # Verify that the correct commands were sent to turn off the laser
    instrument.send_command.assert_any_call(':INST:CHAN 1')
    instrument.send_command.assert_any_call(':MARK:SEL 2')
    instrument.send_command.assert_any_call(':MARK:VOLT:PTOP 0')
    instrument.send_command.assert_any_call(':MARK:VOLT:OFFS 0')
    instrument.send_command.assert_any_call(':MARK OFF')  # Check if the marker was turned off
    instrument.logger.info.assert_called_with("Turning off CH1 Marker 2 (laser control)")  # Logging check
def test_get_ch1_marker2_voltage(instrument):
    # Mock the responses for the send_command queries
    instrument.send_command.return_value = 1.0  # Simulate a return value for PTOP and OFFS

    # Call the method
    low_voltage, high_voltage = instrument.get_ch1_marker2_voltage()

    # Verify the voltage is calculated correctly
    assert low_voltage == 0.5  # (1.0 - 0.5 * 1.0)
    assert high_voltage == 1.5  # (1.0 + 0.5 * 1.0)

    # Verify if the correct commands were issued
    instrument.send_command.assert_any_call(':INST:CHAN 1')
    instrument.send_command.assert_any_call(':MARK:SEL 2')
    instrument.send_command.assert_any_call(':MARK:VOLT:PTOP?')
    instrument.send_command.assert_any_call(':MARK:VOLT:OFFS?')
def test_get_ch1_marker2_voltage_error(instrument):
    # Simulate an exception being thrown when sending commands
    instrument.send_command.side_effect = Exception("Instrument failure")

    # Call the method and check the result
    low_voltage, high_voltage = instrument.get_ch1_marker2_voltage()

    # Assert that the function returns None values on failure
    assert low_voltage is None
    assert high_voltage is None

    # Verify that the error was logged
    instrument.logger.error.assert_called_with("Failed to get CH1 Marker 2 voltage: Instrument failure")
def test_set_function_generator(instrument):
    # Mock the methods involved in the waveform generation
    instrument.send_wfm_to_proteus = MagicMock()
    instrument.configure_sampling_mode = MagicMock(return_value=("DAC_MODE", "interpolation", 1000, 10, [1, 2], [1]))
    instrument.apply_sampling_configuration = MagicMock()

    # Call the function with some sample parameters
    success = instrument.set_function_generator(channel=1, function='SIN', frequency=10e6, voltage=1.0, phase=0.0)

    # Verify if the waveform was successfully sent
    assert success is True
    instrument.send_wfm_to_proteus.assert_called_with(samplingRate=1250000000, channel=1, segment=1, myWfm=MagicMock(), dacRes="DAC_MODE", initialize=True)

    # Verify the logger was called
    instrument.logger.info.assert_called_with("Configuring channel 1: SIN, 10000000.0 Hz, 1.0V, phase=0.0°")
