import React, { useState, useEffect, useContext } from 'react';
import {
    BrowserRouter as Router,
    Routes,
    Route,
    Navigate,
    Outlet,
} from 'react-router-dom';
import {
    Container,
    Typography,
    CircularProgress,
    Alert,
    CssBaseline,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Box,
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import axios from 'axios';
import Navbar from './components/Navbar';
import AuthProvider, { AuthContext } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './components/Auth/Login';
import Signup from './components/Auth/Signup';
import IndicatorSelector from './components/IndicatorSelector';
import CustomDatePicker from './components/DatePicker';
import ResultsDisplay from './components/ResultsDisplay';
import GenerateButton from './components/GenerateButton';
import BacktestButton from './components/BacktestButton';

const darkTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#FFD700', // Golden color
        },
        secondary: {
            main: '#f48fb1',
        },
    },
});

const MainApp = () => {
    const { auth } = useContext(AuthContext);
    console.log('MainApp rendering, auth:', auth);
    const [tickers, setTickers] = useState([]);
    const [selectedIndicator, setSelectedIndicator] = useState('FairValueGap'); // Default to FairValueGap
    const [symbol, setSymbol] = useState('');
    const [interval, setInterval] = useState('1h');
    const [startDate, setStartDate] = useState(new Date(new Date().setDate(1)));
    const [endDate, setEndDate] = useState(new Date());
    const [generateResults, setGenerateResults] = useState({
        plotImage: null,
        metrics: null,
        trades: null
    });
    const [backtestResults, setBacktestResults] = useState({
        plotImage: null,
        metrics: null,
        trades: null
    });
    const [loadingGenerate, setLoadingGenerate] = useState(false);
    const [loadingBacktest, setLoadingBacktest] = useState(false);
    const [errorGenerate, setErrorGenerate] = useState(null);
    const [errorBacktest, setErrorBacktest] = useState(null);

    useEffect(() => {
        const fetchTickers = async () => {
            try {
                const response = await axios.get('http://localhost:8000/tickers', {
                    headers: {
                        Authorization: `Bearer ${auth.token}`,
                    },
                });
                setTickers(response.data.tickers);
                if (response.data.tickers.length > 0) {
                    setSymbol(response.data.tickers[0].Symbol);
                }   
            } catch (err) {
                console.error('Error fetching tickers:', err);
                setErrorGenerate('Failed to fetch tickers');
            }
        };

        if (auth.token) {
            fetchTickers();
        }
    }, [auth.token]);

    const handleGenerate = async () => {
        setLoadingGenerate(true);
        setErrorGenerate(null);

        try {
            const response = await axios.post('http://localhost:8000/generate_plot', {
                indicator: selectedIndicator,
                symbol: symbol,
                interval: interval,
                start_date: startDate.toISOString().split('T')[0],
                end_date: endDate.toISOString().split('T')[0],
            }, {
                headers: {
                    Authorization: `Bearer ${auth.token}`,
                    'Content-Type': 'application/json; charset=utf-8',
                },
            });

            console.log('Generate Response:', response.data);

            setGenerateResults({
                plotImage: response.data.plot_image ? `data:image/png;base64,${response.data.plot_image}` : null,
                metrics: response.data.metrics,
                trades: response.data.closed_trades
            });
        } catch (err) {
            console.error('Generate Error:', err);
            setErrorGenerate(err.response?.data.detail || 'Failed to generate plot');
        } finally {
            setLoadingGenerate(false);
        }
    };

    const handleBacktest = async () => {
        setLoadingBacktest(true);
        setErrorBacktest(null);

        try {
            const response = await axios.post('http://localhost:8000/api/backtest', {
                indicator: selectedIndicator,
                symbol: symbol,
                interval: interval,
                start_date: startDate.toISOString().split('T')[0],
                end_date: endDate.toISOString().split('T')[0],
            }, {
                headers: {
                    Authorization: `Bearer ${auth.token}`,
                    'Content-Type': 'application/json; charset=utf-8',
                },
            });

            console.log('Backtest Response:', response.data);

            setBacktestResults({
                metrics: response.data.metrics,
                trades: response.data.closed_trades
            });
        } catch (err) {
            console.error('Backtest Error:', err);
            setErrorBacktest(err.response?.data.detail || 'Failed to backtest');
        } finally {
            setLoadingBacktest(false);
        }
    };

    const handleIndicatorChange = (newIndicator) => {
        setSelectedIndicator(newIndicator);
        // Clear generate results
        setGenerateResults({
            plotImage: null,
            metrics: null,
            trades: null
        });
        // Clear backtest results
        setBacktestResults({
            plotImage: null,
            metrics: null,
            trades: null
        });
        // Clear any errors
        setErrorGenerate(null);
        setErrorBacktest(null);
    };

    const handleSymbolChange = (newSymbol) => {
        setSymbol(newSymbol);
        // Clear generate results
        setGenerateResults({
            plotImage: null,
            metrics: null,
            trades: null
        });
        // Clear backtest results
        setBacktestResults({
            plotImage: null,
            metrics: null,
            trades: null
        });
        // Clear any errors
        setErrorGenerate(null);
        setErrorBacktest(null);
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
                Welcome, {auth.user?.username} ({auth.user?.tier})
            </Typography>
            <Typography variant="h4" gutterBottom style={{ marginTop: '16px' }}>
                BANG Indicators and Backtests
            </Typography>

            <IndicatorSelector
                selectedIndicator={selectedIndicator}
                setSelectedIndicator={handleIndicatorChange}
            />

            <Box sx={{ marginTop: 2 }}>
                <FormControl fullWidth>
                    <InputLabel id="symbol-label">Select Ticker</InputLabel>
                    <Select
                        labelId="symbol-label"
                        id="symbol"
                        value={symbol}
                        label="Select Ticker"
                        onChange={(e) => handleSymbolChange(e.target.value)}
                    >
                        {tickers.map((ticker) => (
                            <MenuItem key={ticker.Symbol} value={ticker.Symbol}>
                                {ticker.Name} ({ticker.Symbol})
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Box>

            <Box sx={{ marginTop: 2, display: 'flex', gap: 2 }}>
                <FormControl fullWidth>
                    <InputLabel id="interval-label">Select Interval</InputLabel>
                    <Select
                        labelId="interval-label"
                        id="interval"
                        value={interval}
                        label="Select Interval"
                        onChange={(e) => setInterval(e.target.value)}
                    >
                        <MenuItem value="1m">1 Minute</MenuItem>
                        <MenuItem value="5m">5 Minutes</MenuItem>
                        <MenuItem value="15m">15 Minutes</MenuItem>
                        <MenuItem value="30m">30 Minutes</MenuItem>
                        <MenuItem value="1h">1 Hour</MenuItem>
                        <MenuItem value="4h">4 Hours</MenuItem>
                        <MenuItem value="1d">1 Day</MenuItem>
                    </Select>
                </FormControl>
            </Box>

            <Box sx={{ marginTop: 2, display: 'flex', gap: 2 }}>
                <CustomDatePicker
                    startDate={startDate}
                    setStartDate={setStartDate}
                    endDate={endDate}
                    setEndDate={setEndDate}
                />
            </Box>

            <Box sx={{ marginTop: 2, display: 'flex', gap: 2 }}>
                <GenerateButton handleGenerate={handleGenerate} disabled={loadingGenerate} />
                {auth.user && auth.user.tier === 'Pro' && (
                    <BacktestButton handleBacktest={handleBacktest} disabled={loadingBacktest} />
                )}
            </Box>

            <Box sx={{ marginTop: 2 }}>
                {(loadingGenerate || loadingBacktest) && <CircularProgress />}
            </Box>

            <Box sx={{ marginTop: 2 }}>
                {errorGenerate && (
                    <Alert severity="error">{errorGenerate}</Alert>
                )}
                {errorBacktest && (
                    <Alert severity="error">{errorBacktest}</Alert>
                )}
            </Box>

            {/* Display Generate Results */}
            {generateResults.plotImage && (
                <Box sx={{ marginTop: 4, textAlign: 'center' }}>
                    <Typography variant="h5" gutterBottom>
                        Generate Results: {selectedIndicator === 'FairValueGap' ? 'Fair Value Gaps Plot' : 'Williams %R Plot'}
                    </Typography>
                    <img src={generateResults.plotImage} alt={`${selectedIndicator} Generate Plot`} style={{ maxWidth: '100%' }} />
                    {generateResults.metrics && generateResults.trades && (
                        <ResultsDisplay metrics={generateResults.metrics} trades={generateResults.trades} />
                    )}
                </Box>
            )}

            {/* Display Backtest Results */}
            {backtestResults.metrics && backtestResults.trades && (
                <Box sx={{ marginTop: 4, textAlign: 'center' }}>
                    <Typography variant="h5" gutterBottom>
                        Backtest Results: {selectedIndicator}
                    </Typography>
                    <ResultsDisplay metrics={backtestResults.metrics} trades={backtestResults.trades} />
                </Box>
            )}
        </Container>
    );
};

const App = () => {
    return (
        <ThemeProvider theme={darkTheme}>
            <CssBaseline />
            <Router>
                <AuthProvider>
                    <Routes>
                        <Route path="/login" element={<Login />} />
                        <Route path="/signup" element={<Signup />} />
                        <Route element={<ProtectedRoute />}>
                            <Route
                                path="/main"
                                element={
                                    <>
                                        <Navbar />
                                        <MainApp />
                                    </>
                                }
                            />
                        </Route>
                        <Route path="/" element={<Navigate to="/login" replace />} />
                    </Routes>
                </AuthProvider>
            </Router>
        </ThemeProvider>
    );
};

export default App;
