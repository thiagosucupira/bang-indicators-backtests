import React, { useState, useContext } from 'react';
import { TextField, Button, Box, Typography, Alert } from '@mui/material';
import { AuthContext } from '../../context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const { login } = useContext(AuthContext);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await login(username, password);
            navigate('/main');  // Change this from '/' to '/main'
        } catch (err) {
            console.error('Login error:', err);
            setError(err.message || 'Login failed');
        }
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                maxWidth: 300,
                margin: 'auto',
                marginTop: 4,
            }}
        >
            <Typography variant="h4" component="h1" gutterBottom>
                Login
            </Typography>
            {error && <Alert severity="error">{error}</Alert>}
            <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
                <TextField
                    margin="normal"
                    required
                    fullWidth
                    id="username"
                    label="Username"
                    name="username"
                    autoComplete="username"
                    autoFocus
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    InputLabelProps={{ style: { color: '#ffffff' } }}
                    InputProps={{
                        style: {
                            color: '#ffffff',
                            appearance: 'none', // Ensure 'appearance' is used instead of '-moz-appearance'
                        },
                    }}
                />
                <TextField
                    margin="normal"
                    required
                    fullWidth
                    name="password"
                    label="Password"
                    type="password"
                    id="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    InputLabelProps={{ style: { color: '#ffffff' } }}
                    InputProps={{
                        style: {
                            color: '#ffffff',
                            appearance: 'none', // Ensure 'appearance' is used instead of '-moz-appearance'
                        },
                    }}
                />
                <Button
                    type="submit"
                    fullWidth
                    variant="contained"
                    sx={{ mt: 3, mb: 2 }}
                >
                    Login
                </Button>
                <Button 
                    variant="contained" 
                    fullWidth
                    component={Link} 
                    to="/signup" 
                    sx={{ mt: 2 }}
                >
                    Sign Up
                </Button>
            </Box>
        </Box>
    );
};

export default Login;