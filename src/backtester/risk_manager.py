import logging
import configparser

# Configure logging for this module
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Handles risk management aspects such as position sizing, stop-loss, and take-profit orders.
    Parameters are loaded from the project's configuration file.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        Initializes the RiskManager with parameters from the configuration.

        Args:
            config (configparser.ConfigParser): The loaded configuration object.
        """
        self.config = config
        self._load_config()

    def _load_config(self):
        """
        Loads risk management parameters from the 'RISK_MANAGER' section of the config.
        """
        try:
            risk_section = self.config['RISK_MANAGER']
            self.position_sizing_type = risk_section.get('position_sizing_type', 'fixed_capital_percentage').lower()
            self.capital_allocation_percentage = risk_section.getfloat('capital_allocation_percentage', 2.0)
            self.risk_per_trade_percentage = risk_section.getfloat('risk_per_trade_percentage', 1.0)
            self.default_stop_loss_percentage = risk_section.getfloat('default_stop_loss_percentage', 5.0)
            self.risk_reward_ratio = risk_section.getfloat('risk_reward_ratio', 2.0)

            logger.info(f"RiskManager initialized with: "
                        f"Sizing Type: '{self.position_sizing_type}', "
                        f"Capital Allocation: {self.capital_allocation_percentage}%, "
                        f"Risk per Trade: {self.risk_per_trade_percentage}%, "
                        f"Default SL: {self.default_stop_loss_percentage}%, "
                        f"Risk-Reward: {self.risk_reward_ratio}")

        except KeyError:
            logger.error("Missing '[RISK_MANAGER]' section in config.ini. Using default values.")
            self.position_sizing_type = 'fixed_capital_percentage'
            self.capital_allocation_percentage = 2.0
            self.risk_per_trade_percentage = 1.0
            self.default_stop_loss_percentage = 5.0
            self.risk_reward_ratio = 2.0
        except ValueError as e:
            logger.error(f"Error parsing risk manager configuration: {e}. Using default values.")
            self.position_sizing_type = 'fixed_capital_percentage'
            self.capital_allocation_percentage = 2.0
            self.risk_per_trade_percentage = 1.0
            self.default_stop_loss_percentage = 5.0
            self.risk_reward_ratio = 2.0

    def calculate_position_size(self, current_price: float, portfolio_value: float, stop_loss_price: float = None) -> int:
        """
        Calculates the number of shares to trade based on configured position sizing.

        Args:
            current_price (float): The current price of the asset.
            portfolio_value (float): The current total value of the portfolio (cash + assets).
            stop_loss_price (float, optional): The calculated stop-loss price for the trade.
                                               Required if position_sizing_type is 'fixed_risk_amount'.

        Returns:
            int: The calculated number of shares. Returns 0 if calculation is not possible or invalid.
        """
        if current_price <= 0:
            logger.warning(f"Cannot calculate position size with non-positive current price: {current_price}. Returning 0 shares.")
            return 0
        if portfolio_value <= 0:
            logger.warning(f"Cannot calculate position size with non-positive portfolio value: {portfolio_value}. Returning 0 shares.")
            return 0

        num_shares = 0

        if self.position_sizing_type == 'fixed_capital_percentage':
            # Allocate a fixed percentage of the total portfolio value for the trade.
            # In a more advanced backtester, this might be based on available cash,
            # but for simplicity, we use portfolio_value as the base.
            amount_to_allocate = portfolio_value * (self.capital_allocation_percentage / 100)
            num_shares = int(amount_to_allocate / current_price)
            logger.debug(f"Position sizing (Fixed Capital %): Portfolio value ${portfolio_value:.2f}, "
                         f"Allocating {self.capital_allocation_percentage}%, Amount: ${amount_to_allocate:.2f}, "
                         f"Current Price: ${current_price:.2f}, Shares: {num_shares}")

        elif self.position_sizing_type == 'fixed_risk_amount':
            if stop_loss_price is None:
                logger.error("Stop-loss price is required for 'fixed_risk_amount' position sizing. Returning 0 shares.")
                return 0
            
            # Calculate the absolute risk per share (distance from entry to stop-loss)
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share <= 0:
                logger.warning(f"Risk per share is zero or negative ({risk_per_share:.2f}). Cannot size position. Returning 0 shares.")
                return 0

            # Calculate the total risk amount based on portfolio value
            total_risk_amount = portfolio_value * (self.risk_per_trade_percentage / 100)
            
            num_shares = int(total_risk_amount / risk_per_share)
            logger.debug(f"Position sizing (Fixed Risk Amount): Portfolio value ${portfolio_value:.2f}, "
                         f"Risk per trade {self.risk_per_trade_percentage}%, Total risk: ${total_risk_amount:.2f}, "
                         f"Risk per share: ${risk_per_share:.2f}, Shares: {num_shares}")
        else:
            logger.warning(f"Unknown position sizing type: '{self.position_sizing_type}'. Returning 0 shares.")
            return 0
        
        return max(0, num_shares) # Ensure non-negative shares

    def calculate_stop_loss_price(self, entry_price: float, is_long: bool) -> float:
        """
        Calculates the stop-loss price based on a percentage from the entry price.

        Args:
            entry_price (float): The price at which the position was entered.
            is_long (bool): True if it's a long position, False if short.

        Returns:
            float: The calculated stop-loss price. Returns 0.0 if calculation is not possible.
        """
        if entry_price <= 0:
            logger.warning(f"Cannot calculate stop-loss for non-positive entry price: {entry_price}. Returning 0.0.")
            return 0.0

        sl_percentage = self.default_stop_loss_percentage / 100.0

        if is_long:
            stop_loss_price = entry_price * (1 - sl_percentage)
        else: # Short position
            stop_loss_price = entry_price * (1 + sl_percentage)
        
        logger.debug(f"Calculated Stop-Loss for {'long' if is_long else 'short'} position: "
                     f"Entry: ${entry_price:.2f}, SL %: {self.default_stop_loss_percentage}%, SL Price: ${stop_loss_price:.2f}")
        return round(stop_loss_price, 2) # Round to 2 decimal places for prices

    def calculate_take_profit_price(self, entry_price: float, stop_loss_price: float, is_long: bool) -> float:
        """
        Calculates the take-profit price based on the risk-reward ratio.

        Args:
            entry_price (float): The price at which the position was entered.
            stop_loss_price (float): The calculated stop-loss price for the trade.
            is_long (bool): True if it's a long position, False if short.

        Returns:
            float: The calculated take-profit price. Returns 0.0 if calculation is not possible.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning(f"Cannot calculate take-profit for non-positive entry/stop-loss prices: Entry ${entry_price}, SL ${stop_loss_price}. Returning 0.0.")
            return 0.0

        # Calculate the absolute risk amount per share (distance from entry to stop-loss)
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share <= 0:
            logger.warning(f"Risk per share is zero or negative ({risk_per_share:.2f}). Cannot calculate take-profit. Returning 0.0.")
            return 0.0

        # Calculate the target profit per share based on risk-reward ratio
        target_profit_per_share = risk_per_share * self.risk_reward_ratio

        if is_long:
            take_profit_price = entry_price + target_profit_per_share
        else: # Short position
            take_profit_price = entry_price - target_profit_per_share
        
        logger.debug(f"Calculated Take-Profit for {'long' if is_long else 'short'} position: "
                     f"Entry: ${entry_price:.2f}, SL: ${stop_loss_price:.2f}, Risk/Share: ${risk_per_share:.2f}, "
                     f"RR Ratio: {self.risk_reward_ratio}, TP Price: ${take_profit_price:.2f}")
        return round(take_profit_price, 2) # Round to 2 decimal places for prices

if __name__ == '__main__':
    # This block is for testing the RiskManager class in isolation.
    # It will not run when the module is imported by other parts of the backtester.

    # Setup basic logging for standalone test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("--- Testing RiskManager ---")

    # Create a dummy config for testing purposes
    test_config = configparser.ConfigParser()
    test_config['RISK_MANAGER'] = {
        'position_sizing_type': 'fixed_risk_amount',
        'capital_allocation_percentage': '2.0', # Not used for 'fixed_risk_amount'
        'risk_per_trade_percentage': '1.0',
        'default_stop_loss_percentage': '5.0',
        'risk_reward_ratio': '2.0'
    }
    test_config['DEFAULT'] = {} # configparser requires a DEFAULT section if others are present

    # Initialize RiskManager
    risk_manager = RiskManager(test_config)

    # --- Test Case 1: Fixed Capital Percentage Sizing ---
    print("\n--- Test Case: Fixed Capital Percentage Sizing (Configured to 2%) ---")
    test_config['RISK_MANAGER']['position_sizing_type'] = 'fixed_capital_percentage'
    risk_manager = RiskManager(test_config) # Re-initialize with new config
    
    current_price_1 = 100.0
    portfolio_value_1 = 10000.0
    shares_1 = risk_manager.calculate_position_size(current_price_1, portfolio_value_1)
    print(f"Current Price: ${current_price_1:.2f}, Portfolio Value: ${portfolio_value_1:.2f}")
    print(f"Calculated Shares (Fixed Capital %): {shares_1}")
    expected_shares_1 = int((portfolio_value_1 * (risk_manager.capital_allocation_percentage / 100)) / current_price_1)
    assert shares_1 == expected_shares_1, f"Expected {expected_shares_1} shares, got {shares_1}"
    print("Test 1 Passed.")

    # --- Test Case 2: Fixed Risk Amount Sizing (Configured to 1% risk, 5% SL, 1:2 RR) ---
    print("\n--- Test Case: Fixed Risk Amount Sizing (Long Position) ---")
    test_config['RISK_MANAGER']['position_sizing_type'] = 'fixed_risk_amount'
    risk_manager = RiskManager(test_config) # Re-initialize with new config

    entry_price_2 = 100.0
    portfolio_value_2 = 10000.0
    
    # Long position calculations
    is_long_2 = True
    sl_price_2_long = risk_manager.calculate_stop_loss_price(entry_price_2, is_long=is_long_2)
    tp_price_2_long = risk_manager.calculate_take_profit_price(entry_price_2, sl_price_2_long, is_long=is_long_2)
    shares_2_long = risk_manager.calculate_position_size(entry_price_2, portfolio_value_2, sl_price_2_long)

    print(f"Entry Price: ${entry_price_2:.2f}, Portfolio Value: ${portfolio_value_2:.2f}")
    print(f"Long Position - SL Price: ${sl_price_2_long:.2f}, TP Price: ${tp_price_2_long:.2f}")
    print(f"Calculated Shares (Fixed Risk Amount, Long): {shares_2_long}")
    
    # Expected calculations for long:
    # SL % = 5% -> SL Price = 100 * (1 - 0.05) = 95.0
    # Risk per share = 100 - 95 = 5.0
    # Total risk amount = 10000 * 0.01 = 100.0
    # Shares = 100.0 / 5.0 = 20
    assert sl_price_2_long == 95.0, f"Expected SL 95.0, got {sl_price_2_long}"
    assert tp_price_2_long == 110.0, f"Expected TP 110.0, got {tp_price_2_long}" # 100 + (5 * 2) = 110
    assert shares_2_long == 20, f"Expected 20 shares, got {shares_2_long}"
    print("Test 2 Passed.")

    print("\n--- Test Case: Fixed Risk Amount Sizing (Short Position) ---")
    # Short position calculations
    is_long_3 = False
    sl_price_3_short = risk_manager.calculate_stop_loss_price(entry_price_2, is_long=is_long_3)
    tp_price_3_short = risk_manager.calculate_take_profit_price(entry_price_2, sl_price_3_short, is_long=is_long_3)
    shares_3_short = risk_manager.calculate_position_size(entry_price_2, portfolio_value_2, sl_price_3_short)

    print(f"Short Position - SL Price: ${sl_price_3_short:.2f}, TP Price: ${tp_price_3_short:.2f}")
    print(f"Calculated Shares (Fixed Risk Amount, Short): {shares_3_short}")

    # Expected calculations for short:
    # SL % = 5% -> SL Price = 100 * (1 + 0.05) = 105.0
    # Risk per share = 105 - 100 = 5.0
    # Total risk amount = 10000 * 0.01 = 100.0
    # Shares = 100.0 / 5.0 = 20
    assert sl_price_3_short == 105.0, f"Expected SL 105.0, got {sl_price_3_short}"
    assert tp_price_3_short == 90.0, f"Expected TP 90.0, got {tp_price_3_short}" # 100 - (5 * 2) = 90
    assert shares_3_short == 20, f"Expected 20 shares, got {shares_3_short}"
    print("Test 3 Passed.")

    # --- Test Case 4: Invalid Inputs ---
    print("\n--- Test Case: Invalid Inputs ---")
    shares_invalid_price = risk_manager.calculate_position_size(0, portfolio_value_2, sl_price_2_long)
    print(f"Shares with 0 price: {shares_invalid_price}")
    assert shares_invalid_price == 0, f"Expected 0 shares, got {shares_invalid_price}"

    sl_invalid_entry = risk_manager.calculate_stop_loss_price(0, True)
    print(f"SL with 0 entry: {sl_invalid_entry}")
    assert sl_invalid_entry == 0.0, f"Expected SL 0.0, got {sl_invalid_entry}"

    tp_invalid_entry = risk_manager.calculate_take_profit_price(0, sl_price_2_long, True)
    print(f"TP with 0 entry: {tp_invalid_entry}")
    assert tp_invalid_entry == 0.0, f"Expected TP 0.0, got {tp_invalid_entry}"

    tp_invalid_sl = risk_manager.calculate_take_profit_price(entry_price_2, 0, True)
    print(f"TP with 0 SL: {tp_invalid_sl}")
    assert tp_invalid_sl == 0.0, f"Expected TP 0.0, got {tp_invalid_sl}"

    # Test zero risk per share for fixed_risk_amount
    print("\n--- Test Case: Zero Risk per Share ---")
    shares_zero_risk = risk_manager.calculate_position_size(100, portfolio_value_2, 100) # SL at entry
    print(f"Shares with SL at entry price: {shares_zero_risk}")
    assert shares_zero_risk == 0, f"Expected 0 shares, got {shares_zero_risk}"
    print("Test 4 Passed.")

    print("\nAll RiskManager tests completed successfully!")