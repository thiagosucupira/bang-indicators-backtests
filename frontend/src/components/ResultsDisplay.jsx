import React from 'react';
import {
    Typography,
    Paper,
    Grid,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from '@mui/material';

const ResultsDisplay = ({ metrics, trades, decimals = 5 }) => {
    const formatMetric = (num) => {
        return typeof num === 'number' ? num.toFixed(2) : num;
    };

    const formatTradePrice = (num) => {
        return typeof num === 'number' ? num.toFixed(decimals) : num;
    };

    return (
        <div style={{ marginTop: '32px' }}>
            {/* Strategy Metrics */}
            <Typography variant="h5" gutterBottom>
                Strategy Metrics
            </Typography>
            <Paper style={{ padding: '16px' }}>
                <Grid container spacing={2}>
                    {metrics &&
                        Object.keys(metrics).map((key) => (
                            <Grid item xs={12} sm={6} key={key}>
                                <Typography variant="body1">
                                    <strong>{key}:</strong> {formatMetric(metrics[key])}
                                </Typography>
                            </Grid>
                        ))}
                </Grid>
            </Paper>

            {/* Closed Trades */}
            <Typography variant="h5" gutterBottom style={{ marginTop: '32px' }}>
                Closed Trades
            </Typography>
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Entry Time</TableCell>
                            <TableCell>Exit Time</TableCell>
                            <TableCell>Entry Price</TableCell>
                            <TableCell>Exit Price</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Result</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {trades &&
                            trades.map((trade, index) => (
                                <TableRow key={index} hover>
                                    <TableCell>
                                        {new Date(trade.Entry_Time).toLocaleString()}
                                    </TableCell>
                                    <TableCell>
                                        {new Date(trade.Exit_Time).toLocaleString()}
                                    </TableCell>
                                    <TableCell>
                                        {formatTradePrice(trade.Entry_Price)}
                                    </TableCell>
                                    <TableCell>
                                        {formatTradePrice(trade.Exit_Price)}
                                    </TableCell>
                                    <TableCell>
                                        {trade.Type}
                                    </TableCell>
                                    <TableCell
                                        style={{
                                            color: trade.Result === 'Win' ? 'green' : 'red',
                                            fontWeight: 'bold',
                                        }}
                                    >
                                        {trade.Result}
                                    </TableCell>
                                </TableRow>
                            ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </div>
    );
};

export default ResultsDisplay;