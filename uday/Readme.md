### what is get_sample_fpml_data() doing?
    The get_sample_fpml_data() function generates synthetic FPML (Financial Products Markup Language) data for demonstration purposes when no actual FPML file is uploaded or when there's an error connecting to the backend.
    
    Specifically, it:
    
    Creates a date range spanning 2023 (weekly intervals)
    
    Generates a list of sample FX trades with different types (Spot, Forward, Option, Swap)
    
    Assigns random but realistic values for trade details like:
    
    Trade IDs
    Currency pairs (EUR/USD, USD/JPY, etc.)
    Trade amounts
    Exchange rates
    Value dates
    Option strike rates and expiry dates
    Swap near and far leg details
    Adds type-specific fields to each trade (e.g., strike rates for options, near/far leg dates for swaps)
    
    Returns a structured dictionary containing:
    
    The list of generated trades
    A summary with statistics like total trades, distribution by type, and distribution by currency pair
    This function serves as a fallback data source to ensure the app can display meaningful visualizations even without real FPML data, making it useful for development, testing, and demonstration purposes.


### what type of response expected from LLM
- Basic Structure for All Responses:
    ```{
      "trades": [
        {
          "tradeId": "T1001",
          "type": "Spot",
          "tradeDate": "2023-01-03",
          "pair": "EUR/USD",
          "currency1": "EUR",
          "currency2": "USD",
          "amount1": 1000000,
          "amount2": 1086500
          // Additional fields specific to trade type
        }
        // More trades...
      ],
      "summary": {
        "totalTrades": 50,
        "tradesByType": {"Spot": 20, "Forward": 15, "Option": 10, "Swap": 5},
        "tradesByPair": {"EUR/USD": 25, "USD/JPY": 15, "GBP/USD": 10}
      }
    }
    ```
- For "Trade Types" Chart:
  - Requires the type field for each trade in the trades array
- For "Trade Types" Chart:
  - Requires the pair field for each trade in the trades array
- For "Trade Timeline" Chart:
  - Requires tradeDate and type fields for each trade
  - tradeDate should be in ISO format (YYYY-MM-DD)
- For "Exchange Rates" Chart:
  - For Spot and Forward trades, requires:
     ```
    {
      "type": "Spot", // or "Forward"
      "tradeDate": "2023-01-03",
      "pair": "EUR/USD",
      "rate": 1.0865,
      "tradeId": "T1001",
      "amount1": 1000000
    }
    ```
- For "Option Strikes" Chart:
    - For Option trades, requires:
    ```
    {
      "type": "Option",
      "pair": "EUR/USD",
      "strikeRate": 1.0950,
      "expiryDate": "2023-04-03",
      "optionType": "Call", // or "Put"
      "tradeId": "T1003",
      "amount1": 2000000
    }
    ```
- For "Trade Amounts" Chart:
  - Requires currency1, type, and amount1 fields for each trade
- For Swap Trades (used in various charts):
    ```
    {
      "type": "Swap",
      "tradeId": "T1005",
      "pair": "EUR/USD",
      "currency1": "EUR",
      "currency2": "USD",
      "nearLegValueDate": "2023-01-05",
      "farLegValueDate": "2023-04-05",
      "nearLegRate": 1.0870,
      "farLegRate": 1.0920,
      "amount1": 3000000
    }
    ```
- 3D Scatter Plot
  - Key fields: amount1, rate/strikeRate, tradeDate/expiryDate, type, tradeId
    ```
    {
      "trades": [
        {
          "tradeId": "T1001",
          "type": "Spot",
          "amount1": 5000000,
          "rate": 1.0865,
          "tradeDate": "2023-01-15"
        },
        {
          "tradeId": "T1002",
          "type": "Option",
          "amount1": 3000000,
          "strikeRate": 1.0950,
          "expiryDate": "2023-03-15"
        }
      ]
    }
    ```
- Sunburst Chart
  - Key fields: type, pair, currency1, amount1 (for hierarchical path and size)
  ```
  {
    "trades": [
      {
        "type": "Spot",
        "pair": "EUR/USD",
        "currency1": "EUR",
        "amount1": 5000000
      },
      {
        "type": "Forward",
        "pair": "USD/JPY",
        "currency1": "USD",
        "amount1": 3000000
      }
    ]
  }
  ```
- Animated Bubble Chart
  - Key fields: amount1, rate/strikeRate, tradeDate (for animation frames), type, tradeId
  ```
  {
    "trades": [
      {
        "tradeId": "T1001",
        "type": "Spot",
        "amount1": 5000000,
        "rate": 1.0865,
        "tradeDate": "2023-01-15"
      },
      {
        "tradeId": "T1002",
        "type": "Forward",
        "amount1": 3000000,
        "rate": 1.0950,
        "tradeDate": "2023-02-15"
      }
    ]
  }
  ```
- Radar Chart
  - Key fields: type, amount1, rate/strikeRate, tradeId (for grouping and metrics)
  ```
  {
    "trades": [
      {
        "type": "Spot",
        "amount1": 5000000,
        "rate": 1.0865,
        "tradeId": "T1001"
      },
      {
        "type": "Option",
        "amount1": 3000000,
        "strikeRate": 1.0950,
        "tradeId": "T1002"
      }
    ]
  }
  ```
- Network Graph
  - Key fields: currency1, currency2, amount1 (for nodes and edge weights)
  ```
  {
    "trades": [
      {
        "currency1": "EUR",
        "currency2": "USD",
        "amount1": 5000000
      },
      {
        "currency1": "USD",
        "currency2": "JPY",
        "amount1": 3000000
      }
    ]
  }
  ```
- - -  Each visualization adapts to the available fields in the data, with fallbacks when specific fields are missing. The code handles these variations by checking for field existence before using them.

  - - - - - The LLM should process the FPML data and return a JSON response with this structure, containing all the necessary fields for the selected chart type. The application will then use this structured data to generate the appropriate visualization.

### next heading here