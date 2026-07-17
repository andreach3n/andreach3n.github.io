// =========================================================================
// Paper-airplane cursor ✈  (homepage only — imported by index.astro)
//
// The plane glides after the mouse and drops a dashed flight path that
// fades after a couple of seconds. Everything tunable lives in CONFIG.
// (Mouse only; it stays out of the way on touch screens and for
// reduced-motion users.)
// =========================================================================

const CONFIG = {
	ease: 0.16,    // how quickly the plane catches up to the mouse (0..1)
	dashEvery: 16, // px of flight between trail dashes
};

const fine = window.matchMedia('(pointer: fine)').matches;
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (fine && !reduced) init();

function init() {
	document.body.classList.add('plane-cursor');

	const plane = document.createElement('div');
	plane.className = 'plane';
	// The icon's nose points up-right (-45°), so we add 45° when rotating.
	plane.innerHTML = `
		<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
			stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
			<path d="m22 2-7 20-4-9-9-4Z" />
			<path d="M22 2 11 13" />
		</svg>`;
	document.body.appendChild(plane);

	let mx = -100, my = -100;   // mouse (target)
	let px = -100, py = -100;   // plane (eased toward the mouse)
	let angle = -Math.PI / 4;   // keep last heading when idle
	let lastDashX = -100, lastDashY = -100;
	let seen = false;

	window.addEventListener('pointermove', (e) => {
		mx = e.clientX;
		my = e.clientY;
		if (!seen) {
			seen = true;
			px = lastDashX = mx;
			py = lastDashY = my;
		}
	});

	const frame = () => {
		const dx = mx - px, dy = my - py;
		if (Math.hypot(dx, dy) > 0.5) angle = Math.atan2(dy, dx);
		px += dx * CONFIG.ease;
		py += dy * CONFIG.ease;
		plane.style.transform =
			`translate(${px}px, ${py}px) rotate(${angle + Math.PI / 4}rad)`;

		// Drop a dash every ~16px of flight, angled along the path.
		const ddx = px - lastDashX, ddy = py - lastDashY;
		if (seen && Math.hypot(ddx, ddy) > CONFIG.dashEvery) {
			const dash = document.createElement('div');
			dash.className = 'plane-dash';
			dash.style.transform =
				`translate(${lastDashX}px, ${lastDashY}px) rotate(${Math.atan2(ddy, ddx)}rad)`;
			document.body.appendChild(dash);
			dash.addEventListener('animationend', () => dash.remove());
			lastDashX = px;
			lastDashY = py;
		}

		requestAnimationFrame(frame);
	};
	requestAnimationFrame(frame);
}
