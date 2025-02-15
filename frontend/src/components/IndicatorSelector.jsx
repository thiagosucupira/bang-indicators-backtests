import React from 'react';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const IndicatorSelector = ({ selectedIndicator, setSelectedIndicator }) => {
    return (
        <FormControl fullWidth>
            <InputLabel id="indicator-label">Select Indicator</InputLabel>
            <Select
                labelId="indicator-label"
                id="indicator"
                value={selectedIndicator}
                label="Select Indicator"
                onChange={(e) => setSelectedIndicator(e.target.value)}
            >
                <MenuItem value="FairValueGap">Fair Value Gap</MenuItem>
                <MenuItem value="Williams_R">Williams %R</MenuItem>
                <MenuItem value="Crossover">SMA and EMA Crossover</MenuItem>
                <MenuItem value="Breakout">NYO Range Support/Resistance</MenuItem>
                <MenuItem value="Markov">Markov (HMM)</MenuItem>
            </Select>
        </FormControl>
    );
};

export default IndicatorSelector;
