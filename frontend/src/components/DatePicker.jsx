import React from 'react';
import { TextField, Box } from '@mui/material';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

const CustomDatePicker = ({ startDate, setStartDate, endDate, setEndDate }) => {
    return (
        <Box sx={{ display: 'flex', gap: 2, marginTop: 2 }}>
            <DatePicker
                selected={startDate}
                onChange={(date) => setStartDate(date)}
                selectsStart
                startDate={startDate}
                endDate={endDate}
                maxDate={endDate}
                customInput={<TextField label="Start Date" fullWidth />}
                showMonthDropdown
                showYearDropdown
                dropdownMode="select"
            />
            <DatePicker
                selected={endDate}
                onChange={(date) => setEndDate(date)}
                selectsEnd
                startDate={startDate}
                endDate={endDate}
                minDate={startDate}
                maxDate={new Date()}
                customInput={<TextField label="End Date" fullWidth />}
                showMonthDropdown
                showYearDropdown
                dropdownMode="select"
            />
        </Box>
    );
};

export default CustomDatePicker;