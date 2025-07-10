import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Tuple

class OptionPosition:
    def __init__(self, option_type: str, position_type: str, strike: float, premium: float, quantity: int = 1):
        self.option_type = option_type.lower()
        self.position_type = position_type.lower()
        self.strike = strike
        self.premium = premium
        self.quantity = quantity

    def calculate_payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff for a single option position at various underlying prices.

        Args:
            S (np.ndarray): Array of underlying prices at expiration.

        Returns:
            np.ndarray: Array of payoffs for the option position.
        """
        if self.option_type == 'call':
            if self.position_type == 'buy':
                payoff = np.maximum(S - self.strike, 0) - self.premium
            else:  # sell
                payoff = -np.maximum(S - self.strike, 0) + self.premium
        else:  # put
            if self.position_type == 'buy':
                payoff = np.maximum(self.strike - S, 0) - self.premium
            else:  # sell
                payoff = -np.maximum(self.strike - S, 0) + self.premium
        return payoff * self.quantity

def calculate_underlying_payoff(S: np.ndarray, entry_price: float, position: int) -> np.ndarray:
    """
    Calculates the payoff for an underlying stock position.

    Args:
        S (np.ndarray): Array of underlying prices.
        entry_price (float): The average entry price of the underlying.
        position (int): Quantity of underlying shares (positive for long, negative for short).

    Returns:
        np.ndarray: Array of payoffs for the underlying position.
    """
    return position * (S - entry_price)

def find_breakeven_points(S: np.ndarray, payoff: np.ndarray) -> List[float]:
    """
    Finds the breakeven points where the total payoff crosses zero.

    Args:
        S (np.ndarray): Array of underlying prices.
        payoff (np.ndarray): Array of total profit/loss.

    Returns:
        List[float]: A list of breakeven prices.
    """
    # Find indices where the sign of the payoff changes
    crossings = np.where(np.diff(np.sign(payoff)))[0]
    breakeven = []
    for idx in crossings:
        x1, x2 = S[idx], S[idx+1]
        y1, y2 = payoff[idx], payoff[idx+1]
        
        # Linear interpolation to find the exact zero-crossing point
        if y1 != y2: # Avoid division by zero if payoff is flat at zero
            x_interp = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakeven.append(x_interp)
    return breakeven

def analyze_positions(positions: List[OptionPosition], current_price: float) -> Tuple[List[str], List[Dict]]:
    """
    Provides basic AI-driven suggestions based on the current option positions.
    This is a simplified analysis and can be expanded with more sophisticated metrics (e.g., Greeks).

    Args:
        positions (List[OptionPosition]): List of current option positions.
        current_price (float): The current price of the underlying asset.

    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing a list of general suggestions
                                       and a list of detailed suggestions (currently not fully implemented).
    """
    suggestions = []
    detailed_suggestions = [] # Placeholder for more detailed, structured suggestions
    
    if not positions:
        return ["No options positions to analyze. Add some options to get suggestions."], []
    
    # Simple net delta approximation for directional bias
    net_delta = 0
    call_strikes = []
    put_strikes = []
    
    for pos in positions:
        if pos.option_type == 'call':
            call_strikes.append(pos.strike)
            # Very rough delta approximation: 0.5 for ITM, 0 for OTM, but simplified here
            # For more accuracy, a proper options pricing model (e.g., Black-Scholes) would be needed
            if pos.position_type == 'buy':
                net_delta += pos.quantity * 0.5 # Assuming roughly 0.5 delta for simplicity
            else: # sell
                net_delta -= pos.quantity * 0.5
        else: # put
            put_strikes.append(pos.strike)
            if pos.position_type == 'buy':
                net_delta -= pos.quantity * 0.5 # Assuming roughly -0.5 delta for simplicity
            else: # sell
                net_delta += pos.quantity * 0.5
    
    # Provide directional bias feedback
    if abs(net_delta) > 0.5: # Threshold for "strongly" biased
        if net_delta > 0:
            suggestions.append("Your options position has a net bullish bias. Consider adding bearish positions (e.g., buying puts or selling calls) to hedge or balance.")
        else:
            suggestions.append("Your options position has a net bearish bias. Consider adding bullish positions (e.g., buying calls or selling puts) to hedge or balance.")
    else:
        suggestions.append("Your options position appears relatively delta-neutral or balanced. You might focus on volatility strategies or fine-tuning with spreads.")
    
    # Suggest common strategies based on existing strikes
    if call_strikes and put_strikes:
        # Check for potential iron condor setup (selling OTM calls and puts)
        # This is a very simplified check
        if min(call_strikes) > current_price and max(put_strikes) < current_price:
            suggestions.append("You have out-of-the-money (OTM) calls and puts. Consider forming an iron condor by selling OTM calls and puts, and buying further OTM calls and puts for defined risk.")
        
        # Check for straddle/strangle if strikes are close to current price
        if any(abs(s - current_price) < current_price * 0.05 for s in call_strikes) and \
           any(abs(s - current_price) < current_price * 0.05 for s in put_strikes):
            suggestions.append("You have options near the current price. Consider a straddle (buying both call and put at same strike) or strangle (buying OTM call and put) for a volatility play.")

    # General advice
    suggestions.append("To adjust your breakeven points or risk/reward profile, consider:")
    suggestions.append("- **Adding more options:** Use the 'Add New Position' form for individual legs.")
    suggestions.append("- **Building specific spreads:** Use the 'Build Spread' section for common strategies like vertical spreads.")
    suggestions.append("- **Adjusting quantities:** Increase or decrease quantities of existing positions to modify exposure.")
    suggestions.append("- **Hedging with underlying:** Adjust your underlying stock position to alter the overall delta.")

    return suggestions, detailed_suggestions

def main():
    """
    Main function to run the Streamlit application for the Options Strategy Builder.
    """
    st.set_page_config(layout="wide", page_title="Advanced Options Strategy Builder")

    st.title("📊 Himanshu's Advanced Options Strategy Builder")
    st.markdown("""
        This application helps you visualize the payoff diagram of various options strategies,
        including individual options, spreads, and combinations with the underlying asset.
        Add your positions and see the potential profit/loss at expiration across different price points.
    """)

    # Initialize session state variables if they don't exist
    if 'positions' not in st.session_state:
        st.session_state.positions = []
    if 'underlying_position' not in st.session_state:
        st.session_state.underlying_position = 0
    if 'underlying_entry_price' not in st.session_state:
        st.session_state.underlying_entry_price = 100.0

    # Sidebar for inputs: Current Market Data, Add Position, Build Spread, Current Positions
    st.sidebar.header("Inputs & Positions")

    # Current market data input
    current_price = st.sidebar.number_input(
        "Current Underlying Price", 
        min_value=0.01, 
        value=100.0, 
        step=1.0,
        help="The current market price of the underlying asset."
    )
    st.session_state.underlying_entry_price = st.sidebar.number_input(
        "Underlying Entry Price", 
        min_value=0.01, 
        value=st.session_state.underlying_entry_price,
        step=1.0,
        help="Your average purchase price for the underlying asset if you hold shares."
    )

    # Add new individual option position form
    with st.sidebar.expander("➕ Add New Option Position", expanded=True):
        with st.form("position_form", clear_on_submit=True): # Added clear_on_submit for convenience
            col1, col2 = st.columns(2)
            with col1:
                option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type_add")
                strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0, key="strike_add")
            with col2:
                position_type = st.selectbox("Buy/Sell", ["Buy", "Sell"], key="position_type_add")
                premium = st.number_input("Premium", min_value=0.0, value=5.0, step=0.5, key="premium_add")
            quantity = st.number_input("Quantity", min_value=1, value=1, key="quantity_add")
            
            if st.form_submit_button("Add Position"):
                st.session_state.positions.append(
                    OptionPosition(option_type, position_type, strike, premium, quantity))
                st.rerun()

    # Spread builder
    with st.sidebar.expander("🔧 Build Spread", expanded=True):
        with st.form("spread_form", clear_on_submit=True): # Added clear_on_submit for convenience
            spread_type = st.selectbox("Spread Type", ["Call Spread", "Put Spread"], key="spread_type_build")
            col1, col2 = st.columns(2)
            with col1:
                long_strike = st.number_input("Long Strike", value=current_price, step=1.0, key="long_strike_build")
                long_premium = st.number_input("Long Premium", value=5.0, step=0.5, key="long_premium_build")
            with col2:
                # Default short strike based on spread type and current price
                default_short_strike = current_price * 1.1 if spread_type == "Call Spread" else current_price * 0.9
                short_strike = st.number_input("Short Strike", 
                                             value=default_short_strike, 
                                             step=1.0, 
                                             key="short_strike_build")
                short_premium = st.number_input("Short Premium", value=2.0, step=0.5, key="short_premium_build")
            
            if st.form_submit_button("Add Spread"):
                opt_type = "call" if spread_type == "Call Spread" else "put"
                st.session_state.positions.append(OptionPosition(opt_type, "buy", long_strike, long_premium))
                st.session_state.positions.append(OptionPosition(opt_type, "sell", short_strike, short_premium))
                st.rerun()

    # Underlying position
    st.session_state.underlying_position = st.sidebar.number_input(
        "Underlying Shares", 
        value=st.session_state.underlying_position,
        help="Positive for long, negative for short (e.g., 100 for long 100 shares, -50 for short 50 shares)"
    )

    # Current positions display and removal
    with st.sidebar.expander("Current Positions", expanded=True):
        if not st.session_state.positions and st.session_state.underlying_position == 0:
            st.write("No positions added yet. Use the forms above to add options or underlying shares.")
        else:
            if st.session_state.underlying_position != 0:
                st.write(f"📈 Underlying: {st.session_state.underlying_position} shares @ ${st.session_state.underlying_entry_price:.2f}")
            
            # Display options positions with a delete button
            for i, pos in enumerate(st.session_state.positions[:]):
                cols = st.columns([3, 2, 1])
                cols[0].write(f"{pos.quantity} {pos.position_type.capitalize()} {pos.option_type.capitalize()} @ ${pos.strike:.2f}")
                cols[1].write(f"Premium: ${pos.premium:.2f}")
                if cols[2].button("❌ Remove", key=f"del_{i}"):
                    st.session_state.positions.pop(i)
                    st.rerun()

    # Price range selection for the payoff diagram
    st.subheader("📈 Price Range for Payoff Diagram")
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        # Default min price to 98% of current price
        min_price = st.number_input("Minimum Price", value=max(0.0, current_price * 0.98), step=0.5)
    with col_range2:
        # Default max price to 102% of current price
        max_price = st.number_input("Maximum Price", value=current_price * 1.02, step=0.5)
    
    # Ensure min_price is less than max_price
    if min_price >= max_price:
        st.error("Minimum Price must be less than Maximum Price. Please adjust the range.")
        return # Stop execution if range is invalid

    # Calculate payoffs for plotting
    S = np.linspace(min_price, max_price, 500) # 500 points for smooth curve
    total_payoff = np.zeros_like(S)
    position_payoffs = [] # To store individual position payoffs for plotting

    # Add underlying payoff to total if applicable
    if st.session_state.underlying_position != 0:
        underlying_payoff = calculate_underlying_payoff(S, st.session_state.underlying_entry_price, st.session_state.underlying_position)
        total_payoff += underlying_payoff
        position_payoffs.append(("Underlying Shares", underlying_payoff))
    
    # Add options payoffs to total
    for i, pos in enumerate(st.session_state.positions):
        payoff = pos.calculate_payoff(S)
        total_payoff += payoff
        position_payoffs.append((f"{pos.quantity} {pos.position_type.capitalize()} {pos.option_type.capitalize()} @ {pos.strike}", payoff))
    
    # Calculate key metrics
    breakeven = find_breakeven_points(S, total_payoff)
    max_payoff, min_payoff = total_payoff.max(), total_payoff.min()
    
    # Find the underlying price at which max/min payoff occurs
    max_price_payoff = S[total_payoff.argmax()]
    min_price_payoff = S[total_payoff.argmin()]

    # Create the Plotly figure
    fig = go.Figure()

    # Add individual position payoff lines
    for name, payoff in position_payoffs:
        fig.add_trace(go.Scatter(x=S, y=payoff, mode='lines', name=name, line=dict(width=1), opacity=0.6,
                                 hovertemplate=f'Price: %{{x:.2f}}<br>Payoff: %{{y:.2f}}<extra>{name}</extra>'))
    
    # Add total payoff line (bold black)
    fig.add_trace(go.Scatter(x=S, y=total_payoff, mode='lines', name='Total Payoff', line=dict(color='black', width=3),
                             hovertemplate='Price: %{x:.2f}<br>Total Payoff: %{y:.2f}<extra></extra>'))
    
    # Add breakeven points as markers
    if breakeven:
        for i, be in enumerate(breakeven):
            fig.add_trace(go.Scatter(
                x=[be], y=[0], mode='markers', marker=dict(size=12, color='purple', symbol='diamond'),
                name=f'Breakeven {i+1}', hovertemplate=f'Breakeven: %{{x:.2f}}<extra></extra>'
            ))
    
    # Add Max Profit point
    fig.add_trace(go.Scatter(
        x=[max_price_payoff], y=[max_payoff], mode='markers', marker=dict(size=12, color='green', symbol='star'),
        name='Max Profit', hovertemplate=f'Max Profit: %{{y:.2f}} @ %{{x:.2f}}<extra></extra>'
    ))
    
    # Add Max Loss point
    fig.add_trace(go.Scatter(
        x=[min_price_payoff], y=[min_payoff], mode='markers', marker=dict(size=12, color='red', symbol='x'),
        name='Max Loss', hovertemplate=f'Max Loss: %{{y:.2f}} @ %{{x:.2f}}<extra></extra>'
    ))
    
    # Add vertical line for current price
    fig.add_shape(type='line', x0=current_price, y0=min_payoff, x1=current_price, y1=max_payoff,
                 line=dict(color='blue', dash='dash'), name='Current Price')
    
    # Add horizontal line for zero profit/loss
    fig.add_shape(type='line', x0=min_price, y0=0, x1=max_price, y1=0, line=dict(color='gray', width=1))
    
    # Update layout for better visualization
    fig.update_layout(
        title="Strategy Payoff Diagram",
        xaxis_title="Underlying Price at Expiration",
        yaxis_title="Profit/Loss",
        hovermode='x unified', # Shows all traces' data at a given x-coordinate
        showlegend=True,
        height=600,
        template="plotly_white" # Clean white background
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics display
    st.subheader("Key Metrics")
    cols = st.columns(5)
    cols[0].metric("Current Price", f"${current_price:.2f}")
    cols[1].metric("Underlying Entry Price", f"${st.session_state.underlying_entry_price:.2f}")
    cols[2].metric("Max Profit", f"${max_payoff:.2f}", f"at ${max_price_payoff:.2f}")
    cols[3].metric("Max Loss", f"${min_payoff:.2f}", f"at ${min_price_payoff:.2f}")
    
    # Display all breakeven points clearly
    breakeven_str = ", ".join(f"${be:.2f}" for be in breakeven) if breakeven else "None"
    cols[4].metric("Breakeven Point(s)", breakeven_str)
    
    # AI Recommendations
    st.subheader("💡 AI Strategy Recommendations")
    suggestions, _ = analyze_positions(st.session_state.positions, current_price)
    
    with st.expander("Recommendations to Adjust Your Strategy", expanded=True):
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
        
        st.markdown("\n**Quick Spread Builders:**")
        spread_cols = st.columns(2)
        with spread_cols[0]:
            if st.button("Add Bull Call Spread (near current price)"):
                st.session_state.positions.append(OptionPosition("call", "buy", current_price, 5.0))
                st.session_state.positions.append(OptionPosition("call", "sell", current_price + 5, 2.0)) # Example spread
                st.rerun()
        with spread_cols[1]:
            if st.button("Add Bear Put Spread (near current price)"):
                st.session_state.positions.append(OptionPosition("put", "buy", current_price, 5.0))
                st.session_state.positions.append(OptionPosition("put", "sell", current_price - 5, 2.0)) # Example spread
                st.rerun()

if __name__ == "__main__":
    main()
