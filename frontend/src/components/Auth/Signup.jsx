import React, { useState, useContext } from 'react';
import { TextField, Button, Box, Typography, Alert, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { AuthContext } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const Signup = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [tier, setTier] = useState('Noob');
    const [error, setError] = useState('');
    const { signup } = useContext(AuthContext);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await signup(username, email, password, tier);
            navigate('/login');
        } catch (err) {
            setError(err.response?.data.detail || 'Signup failed');
        }
    };

    return (
        <Box sx={{ maxWidth: 400, margin: 'auto', marginTop: 8, padding: 3, backgroundColor: '#2c2c2c', borderRadius: 2 }}>
            <Typography variant="h5" gutterBottom>
                Sign Up
            </Typography>
            {error && <Alert severity="error">{error}</Alert>}
            <form onSubmit={handleSubmit}>
                <TextField
                    label="Username"
                    variant="outlined"
                    fullWidth
                    required
                    margin="normal"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    InputLabelProps={{ style: { color: '#ffffff' } }}
                    InputProps={{ style: { color: '#ffffff' } }}
                />
                <TextField
                    label="Email"
                    type="email"
                    variant="outlined"
                    fullWidth
                    required
                    margin="normal"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    InputLabelProps={{ style: { color: '#ffffff' } }}
                    InputProps={{ style: { color: '#ffffff' } }}
                />
                <TextField
                    label="Password"
                    type="password"
                    variant="outlined"
                    fullWidth
                    required
                    margin="normal"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    InputLabelProps={{ style: { color: '#ffffff' } }}
                    InputProps={{ style: { color: '#ffffff' } }}
                />
                <FormControl fullWidth margin="normal">
                    <InputLabel id="tier-label" style={{ color: '#ffffff' }}>User Tier</InputLabel>
                    <Select
                        labelId="tier-label"
                        id="tier"
                        value={tier}
                        label="User Tier"
                        onChange={(e) => setTier(e.target.value)}
                        style={{ color: '#ffffff' }}
                    >
                        <MenuItem value="Noob">Noob</MenuItem>
                        <MenuItem value="Pro">Pro</MenuItem>
                    </Select>
                </FormControl>
                <Button
                    type="submit"
                    variant="contained"
                    fullWidth
                    sx={{
                        background: 'linear-gradient(45deg, #FFD700 30%, #FFC107 90%)',
                        color: '#000',
                        borderRadius: '12px',
                        boxShadow: '0 3px 5px 2px rgba(255, 215, 0, .3)',
                        fontWeight: 'bold',
                        height: '50px',
                        marginTop: 2,
                        '&:hover': {
                            background: 'linear-gradient(45deg, #FFC107 30%, #FFD700 90%)',
                        },
                        transition: 'background 0.3s ease',
                    }}
                >
                    Sign Up
                </Button>
            </form>
        </Box>
    );
};

export default Signup;